#!/usr/bin/env python
import json
import asyncio
import random
from collections import defaultdict
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm.asyncio import tqdm

from .constants import LANGUAGES, SIGNATURES, STEREOTYPES, ANALYSIS_PROMPTS, FINAL_SYNTHESIS_PROMPT, SINGLE_PASS_PROMPT
from .utils import extract_prediction
from .llm_api import get_model_response_with_retry, MODEL_NAME
from .evaluation import calculate_metrics


async def get_analysis(client: OpenAI, prompt: str, text: str, analysis_type: str, seed: Optional[int]) -> Dict[str, Any]:
    """Get analysis from the model, using a retry mechanism."""
    messages = [{"role": "user", "content": prompt.format(text=text)}]
    result = await get_model_response_with_retry(
        client=client,
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        call_description=analysis_type,
        seed=seed
    )
    return {
        "type": analysis_type,
        "analysis": result["content"],
        "success": result["success"]
    }


async def process_single_entry(client: OpenAI, entry: Dict[str, Any], condition: str, mode: str, seed: Optional[int], resynthesize_only: bool = False, ablation_exclude: List[str] = []) -> Dict[str, Any]:
    """Process a single entry via agentic (multi-step) or single-pass baseline, with optional text manipulations."""
    text = entry["text"]
    original_lang = entry.get("language")

    if not resynthesize_only:
        # In a normal run, apply modifications based on the run's condition.
        entry["condition"] = condition
        # Signature handling
        if condition in ['adversarial', 'supportive']:
            if original_lang and original_lang in SIGNATURES:
                if condition == 'adversarial':
                    adversarial_langs = [lang for lang in SIGNATURES if lang != original_lang]
                    if adversarial_langs:
                        chosen_lang = random.choice(adversarial_langs)
                        signature = SIGNATURES[chosen_lang]
                        text += signature
                        entry["signature_added"] = f"adversarial ({chosen_lang})"
                elif condition == 'supportive':
                    signature = SIGNATURES[original_lang]
                    text += signature
                    entry["signature_added"] = f"supportive ({original_lang})"

        # Stereotype handling
        if condition in ['stereotype_misleading', 'stereotype_support']:
            if original_lang and original_lang in STEREOTYPES:
                if condition == 'stereotype_misleading':
                    misleading_langs = [lang for lang in STEREOTYPES if lang != original_lang]
                    if misleading_langs:
                        chosen_lang = random.choice(misleading_langs)
                        stereotype_text = STEREOTYPES[chosen_lang]
                        text += f"\n\n{stereotype_text}"
                        entry["stereotype_added"] = f"misleading ({chosen_lang})"
                elif condition == 'stereotype_support':
                    stereotype_text = STEREOTYPES[original_lang]
                    text += f"\n\n{stereotype_text}"
                    entry["stereotype_added"] = f"supportive ({original_lang})"

    if mode == 'single_pass':
        # Single-pass baseline
        messages = [{"role": "user", "content": SINGLE_PASS_PROMPT.format(text=text)}]
        result = await get_model_response_with_retry(
            client=client,
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            call_description="single_pass",
            seed=seed
        )
        synthesis = result["content"]
        processing_success = result["success"]
        predicted_language = extract_prediction(synthesis) if processing_success else "Unknown"
        return {
            **entry,
            "analyses": [],
            "synthesis": synthesis,
            "predicted": predicted_language,
            "processing_success": processing_success
        }

    # Agentic (multi-step) mode
    analyses = []
    if resynthesize_only:
        if 'analyses' in entry and isinstance(entry.get('analyses'), list) and all(a.get("success") for a in entry['analyses']):
            analyses = entry['analyses']
        else:
            return {
                **entry,
                "analyses": entry.get('analyses', []),
                "synthesis": "Failed: --resynthesize_only was specified but no valid 'analyses' field found in input.",
                "predicted": "Unknown",
                "processing_success": False
            }
    
    if not analyses:
        tasks = [
            get_analysis(client, ANALYSIS_PROMPTS["syntax"], text, "syntax", seed),
            get_analysis(client, ANALYSIS_PROMPTS["lexical"], text, "lexical", seed),
            get_analysis(client, ANALYSIS_PROMPTS["idiomatic"], text, "idiomatic", seed)
        ]
        analyses = await asyncio.gather(*tasks)

    failed_analyses = [a for a in analyses if not a["success"]]
    if failed_analyses:
        return {
            **entry,
            "analyses": analyses,
            "synthesis": f"Failed to complete all analyses. Errors: {[a['analysis'] for a in failed_analyses]}",
            "processing_success": False
        }

    # Dynamically construct the prompt from the available, non-excluded analyses.
    analysis_template_parts = []
    analysis_types = ["syntax", "lexical", "idiomatic"]
    
    for analysis_type in analysis_types:
        if analysis_type not in ablation_exclude:
            # Find the corresponding analysis, default to an empty string if not found.
            analysis_text = next((a["analysis"] for a in analyses if a.get("type") == analysis_type), "")
            if analysis_text:
                analysis_template_parts.append(f"{analysis_type.title()} Analysis:\n{analysis_text}")

    analysis_template = "\n\n".join(analysis_template_parts)

    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(
        analysis=analysis_template,
    )
    synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
    synthesis_result = await get_model_response_with_retry(
        client=client,
        model=MODEL_NAME,
        messages=synthesis_messages,
        temperature=0.7,
        max_tokens=2048,
        seed=seed,
        call_description="synthesis"
    )

    synthesis = synthesis_result["content"]
    processing_success = synthesis_result["success"]
    predicted_language = extract_prediction(synthesis) if processing_success else "Unknown"

    return {
        **entry,
        "analyses": analyses,
        "synthesis": synthesis,
        "predicted": predicted_language,
        "processing_success": processing_success
    }


async def process_dataset(client: OpenAI, input_file: str, output_file: str, limit: int, max_concurrent: int, condition: str, mode: str, seed: Optional[int], resynthesize_only: bool = False, ablation_exclude: List[str] = []):
    """Process the entire dataset through the analysis pipeline."""
    with open(input_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Set seeds for reproducibility of sampling/perturbations in this run
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if limit:
        print(f"Balancing dataset for a limit of {limit} entries.")
        entries_by_lang = defaultdict(list)
        for entry in entries:
            if 'language' in entry and entry['language'] in LANGUAGES:
                entries_by_lang[entry['language']].append(entry)

        samples_per_lang = limit // len(LANGUAGES)
        balanced_entries = []
        for lang in LANGUAGES:
            lang_entries = entries_by_lang.get(lang, [])
            num_to_sample = min(samples_per_lang, len(lang_entries))
            if len(lang_entries) < samples_per_lang:
                print(f"Warning: Not enough samples for {lang}. Found {len(lang_entries)}, need {samples_per_lang}. Taking all available.")
            if num_to_sample > 0:
                balanced_entries.extend(random.sample(lang_entries, num_to_sample))

        random.shuffle(balanced_entries)
        entries = balanced_entries
        print(f"Processing a balanced sample of {len(entries)} entries.")
    else:
        print(f"Loaded {len(entries)} entries from {input_file}")

    if condition != 'baseline':
        print(f"Condition '{condition}' enabled.")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(entry):
        async with semaphore:
            return await process_single_entry(client, entry, condition, mode, seed, resynthesize_only, ablation_exclude)

    tasks = [process_with_semaphore(entry) for entry in entries]
    results = await tqdm.gather(*tasks, desc=f"Processing ({condition})")

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    successful = sum(1 for r in results if r["processing_success"])
    print(f"\nRun for condition '{condition}' complete!")
    print(f"Successfully processed: {successful}/{len(results)} entries")
    print(f"Results saved to: {output_file}")

    # --- Evaluation ---
    print("\n--- Evaluation Metrics for this Run ---")
    return calculate_metrics(results)
