#!/usr/bin/env python3
from typing import Dict

# Languages in scope
LANGUAGES = ["French", "Spanish", "Italian", "German"]

# Signatures that can be appended for supportive/adversarial tests
SIGNATURES = {
    'Spanish': '\n\nBest regards,\nMaría García\nMadrid Language School\nCalle de Alcalá 45\n28014 Madrid, Spain',
    'German': '\n\nBest regards,\nJohannes Müller\nBerlin English Institute\nKurfürstendamm 123\n10711 Berlin, Germany',
    'Italian': '\n\nBest regards,\nGiulia Rossi\nMilan English Academy\nVia Monte Napoleone 18\n20121 Milan, Italy',
    'French': '\n\nBest regards,\nLucas Dubois\nParis Language Center\n10 Rue de Rivoli\n75001 Paris, France',
}

# Stereotypes that can be appended for stereotype_* tests
STEREOTYPES = {
    'Spanish': "A fun fact about me: A short nap after lunch is essential for me. My culture even has a word for it: siesta.",
    'Italian': "A fun fact about me: I usually start my day with a quick espresso standing at the bar. And please, no cappuccino in the afternoon!",
    'French': "A fun fact about me: Where I'm from, there's nothing a fresh baguette and a street protest can't fix!",
    'German': "A fun fact about me: I love using bottle-return machines and driving on roads with no speed limit.",
    }

# Prompts for the agentic multi-step pipeline
ANALYSIS_PROMPTS: Dict[str, str] = {
    "syntax": """You are a language expert. Your task is to analyze the following L2 English text exclusively for syntactic errors. 
    The other experts already cover lexical and idiomatic errors on the word level. 
    Focus on grammatical rules like word order, subject-verb agreement, clause structure, tense usage, and modifier placement.,

For each syntactic error identified, include:
1. The `error_type` (e.g., "Incorrect word order", "Subject-verb disagreement").
2. A minimal `explanation` of the grammatical problem.
3. The specific `phrase` (e.g., 3-5 words) where the error occurs.

Do not provide any cultural analysis and references in your error explanations.
Return the output as a JSON array. If no syntactic errors are found, return an empty array.

Text: {text}
""",
    "lexical": """You are a language expert. Your task is to analyze the following L2 English text exclusively for lexical errors.

Focus on identifying and explaining lexical errors where a word is:
- Spelled incorrectly (e.g., false cognates such as "addresse" instead of "address")
- A malapropism (e.g., "illicit" instead of "elicit")
- A false cognate (e.g., "sensible" instead of "sensitive")

For each error, include the `word` containing the lexical error, the `error_type`, and a minimal `explanation`.
Do not provide any cultural analysis and references in your error explanations.
Return the output as a JSON array. If no lexical errors are found, return an empty array.

Text: {text}
""",

    "idiomatic": """You are a language expert. Your task is to analyze the following L2 English text exclusively for idiomatic errors.
    The other experts already cover grammatical and lexical errors. 
Focus on identifying incorrect, awkward, or misused multi-word expressions and figurative expressions. Do not attribute them to a source language yet.
These are typically phrases where the overall meaning is not deducible from the literal meanings of the individual words. Pay attention to:
    - Potential mistranslations or literal translations
    - Violations of common idiomatic expressions in standard English (e.g., "heavy rain" instead of "strong rain").

For each error, include the problematic `expression`, the `error_type`, and a minimal `explanation` of why it's an idiomatic error.
Do not provide any cultural analysis and references in your error explanations.
Return the output as a JSON array. If no idiomatic errors are found, return an empty array.

Text: {text}
"""
}

FINAL_SYNTHESIS_PROMPT = """You are a forensic linguistics expert that reads texts written by non-native authors to identify their native language.
You will be given three different analyses of the text. Analyze the input and identify the native language of the author as one of the following: French, Spanish, Italian, German. 
Use clues such as spelling errors, word choice, and grammatical errors to decide.

{analysis}

Disregard any contextual information, such as names, addresses, institutions, locations, or cultural references in the text.
Provide your analysis in the JSON format.
"""

# Single-pass baseline prompt
SINGLE_PASS_PROMPT = """You are a forensic linguistics expert that reads texts written by non-native authors in order to identify their native language.
Analyze each text and identify the native language of the author as one of the following: French, Spanish, Italian, German. 
Use clues such as spelling errors, word choice, syntactic patterns, and grammatical errors to decide.
Disregard any contextual information, such as names, addresses, institutions, locations, or cultural references in the text.
Provide your analysis in the JSON format.

Text:
{text}
"""