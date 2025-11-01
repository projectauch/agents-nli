import re
from src.constants import LANGUAGES

def extract_prediction(synthesis: str) -> str:
    """
    Extracts the predicted language from the synthesis text.
    It first looks for a structured JSON-like key "native_language" and extracts its value.
    If not found, it falls back to finding the last mentioned language in the text.
    """
    # Prioritize structured JSON-like answer
    match = re.search(r'"native_language"\s*:\s*"([^"]+)"', synthesis, re.IGNORECASE)
    if match:
        predicted_lang_candidate = match.group(1).strip()
        # Check if the found language is one of the valid languages
        for lang in LANGUAGES:
            if lang.lower() == predicted_lang_candidate.lower():
                return lang  # Return with correct capitalization

    # Fallback to searching for the last mentioned language in the whole text
    lower_synthesis = synthesis.lower()
    last_pos = -1
    predicted_lang = "Unknown"
    for lang in LANGUAGES:
        pos = lower_synthesis.rfind(lang.lower())
        if pos > last_pos:
            last_pos = pos
            predicted_lang = lang
    return predicted_lang
