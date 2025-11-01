import numpy as np
from pathlib import Path
from typing import Dict

def generate_ablation_report(metrics_data: Dict, base_output_name: str):
    """Generates a markdown report summarizing the results of the ablation study."""
    report_path = Path("runs") / f"{Path(base_output_name).stem}_ablation_report.md"
    with open(report_path, 'w') as f:
        f.write("# Ablation Study Report\n\n")
        f.write(f"Generated for base file: `{base_output_name}`\n\n")

        if 'keep_one_in' in metrics_data:
            f.write("## Keep-One-In Study (Individual Component Power)\n\n")
            f.write("| Configuration         | Avg Accuracy        | Avg F1-Score        |\n")
            f.write("|-----------------------|---------------------|---------------------|\n")
            for config_name, data in sorted(metrics_data['keep_one_in'].items(), key=lambda item: -np.mean(item[1].get('f1s', [0]))):
                if not data['accuracies']:
                    f.write(f"| {config_name.replace('_', ' ').title():<21} | N/A                 | N/A                 |\n")
                    continue
                acc_mean = np.mean(data['accuracies'])
                acc_std = np.std(data['accuracies'])
                f1_mean = np.mean(data['f1s'])
                f1_std = np.std(data['f1s'])
                acc_str = f"{acc_mean:.4f} \u00B1 {acc_std:.4f}"
                f1_str = f"{f1_mean:.4f} \u00B1 {f1_std:.4f}"
                f.write(f"| {config_name.replace('_', ' ').title():<21} | {acc_str:<19} | {f1_str:<19} |\n")
            f.write("\n")

    print(f"\nAblation report generated at: {report_path}")

def generate_report(metrics_data: Dict, base_output_name: str):
    """Generates a markdown report summarizing the results of all experiments."""
    report_path = Path("runs") / f"{Path(base_output_name).stem}_report.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment Report\n\n")
        f.write(f"Generated for base output: `{base_output_name}`\n\n")
        f.write("| Condition   | Avg Accuracy        | Avg F1-Score        |\n")
        f.write("|-------------|---------------------|---------------------|\n")
        for condition, data in metrics_data.items():
            if not data['accuracies']:
                f.write(f"| {condition.title():<11} | N/A                 | N/A                 |\n")
                continue
            acc_mean = np.mean(data['accuracies'])
            acc_std = np.std(data['accuracies'])
            f1_mean = np.mean(data['f1s'])
            f1_std = np.std(data['f1s'])
            acc_str = f"{acc_mean:.4f} \u00B1 {acc_std:.4f}"
            f1_str = f"{f1_mean:.4f} \u00B1 {f1_std:.4f}"
            f.write(f"| {condition.title():<11} | {acc_str:<19} | {f1_str:<19} |\n")
    print(f"\nFull report generated at: {report_path}")
