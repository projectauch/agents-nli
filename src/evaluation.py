from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.constants import LANGUAGES

def calculate_metrics(results: List[dict]):
    """Calculates accuracy and F1-score from a list of results."""
    y_true, y_pred = [], []
    for r in results:
        if r.get("processing_success") and "language" in r:
            y_true.append(r["language"])
            y_pred.append(r["predicted"])

    if not y_true:
        print("Could not perform evaluation: No successful results with ground truth labels found.")
        return None, None, [], []

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', labels=sorted(LANGUAGES), zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")

    return accuracy, f1, y_true, y_pred
