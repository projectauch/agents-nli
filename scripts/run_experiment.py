#!/usr/bin/env python
import asyncio
import argparse
import random
import os
from pathlib import Path
from openai import OpenAI

from src.reporting import generate_ablation_report, generate_report
from src.experiment import process_dataset

# Load API key from environment variable for security
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

async def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set the environment variable before running the script.")
        return

    parser = argparse.ArgumentParser(description="Run agentic vs single-pass analysis on a text dataset.")
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("output_file", help="Base name for the output JSONL file(s).")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process from the input file.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run each experiment.")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Maximum number of concurrent API requests.")
    parser.add_argument("--condition", choices=['baseline', 'adversarial', 'supportive', 'stereotype_misleading', 'stereotype_support'], default='baseline', help="Specify a single condition to run.")
    parser.add_argument("--mode", choices=['agentic', 'single_pass'], default='single_pass', help="Inference mode: multi-step agentic or single-pass baseline.")
    parser.add_argument("--full_report", action="store_true", help="Run all five conditions and generate a report.")
    parser.add_argument("--stereotype_report", action="store_true", help="Run stereotype conditions (with baseline) and generate a report.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility. Run i uses seed=seed+i. If not set, random seeds are used per run.")
    parser.add_argument("--resynthesize_only", action="store_true", help="If specified, reuse existing analyses from the input file and only re-run the synthesis step.")
    parser.add_argument("--ablation_exclude", nargs='*', choices=['syntax', 'lexical', 'idiomatic'], default=[], help="For a single run, specify analysis types to exclude. Requires --resynthesize_only.")
    parser.add_argument("--run_ablation_study", action="store_true", help="Run a full ablation study (full, w/o syntax, w/o lexical, w/o idiomatic) and generate a report. Requires --resynthesize_only.")
    args = parser.parse_args()

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found."); return

    runs_dir = Path("runs"); runs_dir.mkdir(exist_ok=True)

    if args.run_ablation_study:
        if not args.resynthesize_only:
            print("Error: --run_ablation_study requires --resynthesize_only to be set.")
            return

        print("--- Ablation Study Mode Enabled ---")
        keep_one_in_configs = {
            "full_workflow": [],
            "syntax_only": ["lexical", "idiomatic"],
            "lexical_only": ["syntax", "idiomatic"],
            "idiomatic_only": ["syntax", "lexical"],
        }

        ablation_metrics = {
            "keep_one_in": {name: {'accuracies': [], 'f1s': []} for name in keep_one_in_configs},
        }

        # Run the "Keep-One-In" study
        print("\n" + "#"*10 + " Starting Study: Keep-One-In " + "#"*10)
        for name, excluded_analyses in keep_one_in_configs.items():
            print(f"\n{'='*20} Running Ablation Config: {name.upper()} {'='*20}")
            if args.seed is not None:
                seeds_for_runs = [args.seed + i for i in range(args.runs)]
            else:
                seeds_for_runs = [42 + i for i in range(args.runs)]

            for i in range(args.runs):
                seed_i = seeds_for_runs[i]
                print(f"\n--- Starting Run {i + 1}/{args.runs} for {name} (seed={seed_i}) ---")
                base_name, suffix = Path(args.output_file).stem, Path(args.output_file).suffix
                run_output_file = runs_dir / f"{base_name}_{name}_run_{i+1}{suffix}"
                
                accuracy, f1, _, _ = await process_dataset(
                    client, args.input_file, str(run_output_file), args.limit, 
                    args.max_concurrent, 'baseline', 'agentic', seed_i, 
                    resynthesize_only=True, ablation_exclude=excluded_analyses
                )
                if accuracy is not None and f1 is not None:
                    ablation_metrics["keep_one_in"][name]['accuracies'].append(accuracy)
                    ablation_metrics["keep_one_in"][name]['f1s'].append(f1)

        generate_ablation_report(ablation_metrics, args.output_file)
        return # End execution after study

    if args.full_report:
        conditions_to_run = ['baseline', 'adversarial', 'supportive', 'stereotype_misleading', 'stereotype_support']
    elif args.stereotype_report:
        conditions_to_run = ['stereotype_misleading', 'stereotype_support']
    else:
        conditions_to_run = [args.condition]

    all_metrics = {condition: {'accuracies': [], 'f1s': []} for condition in conditions_to_run}

    # Prepare per-run seeds
    if args.seed is not None:
        seeds_for_runs = [args.seed + i for i in range(args.runs)]
    else:
        seeds_for_runs = [random.randrange(1, 100) for _ in range(args.runs)]

    for condition in conditions_to_run:
        print(f"\n{'='*20} Running Condition: {condition.upper()} {'='*20}")
        for i in range(args.runs):
            seed_i = seeds_for_runs[i]
            print(f"\n--- Starting Run {i + 1}/{args.runs} for {condition} (seed={seed_i}) ---")
            base_name, suffix = Path(args.output_file).stem, Path(args.output_file).suffix
            run_output_file = runs_dir / f"{base_name}_{condition}_run_{i+1}{suffix}"

            accuracy, f1, _, _ = await process_dataset(
                client, args.input_file, str(run_output_file), args.limit, args.max_concurrent, condition, args.mode, seed_i, args.resynthesize_only, args.ablation_exclude
            )

            if accuracy is not None and f1 is not None:
                all_metrics[condition]['accuracies'].append(accuracy)
                all_metrics[condition]['f1s'].append(f1)

    if args.full_report or args.stereotype_report or len(conditions_to_run) > 1:
        generate_report(all_metrics, args.output_file)


if __name__ == "__main__":
    asyncio.run(main())