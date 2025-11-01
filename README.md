# Native Language Identification through Agentic Decomposition

This repository contains the code and instructions to replicate the experiments from the paper "Robust Native Language Identification through Agentic Decomposition" at EMNLP 2025.

## Datasets

This research utilizes two main datasets: TOEFL11 and the Write & Improve 2024 corpus.

### TOEFL11 Dataset

The TOEFL11 dataset is a corpus of essays from learners of English as a foreign language. The `toefl4.jsonl` file used in this project is a subset of this corpus.

*   **Access:** To access the original dataset, you will need to go through the official ETS request process. You can find more information here: [https://catalog.ldc.upenn.edu/LDC2014T06](https://catalog.ldc.upenn.edu/LDC2014T06)

### Write & Improve 2024

The Write & Improve (W&I) corpus contains texts written by learners of English from across the world, which have been annotated. The `write_and_improve_es_de_it_fr_100.jsonl` file is a subset of this corpus.

*   **Access:** The W&I corpus is available through Cambridge University. More information can be found at: [https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024](https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024)


## Setup

1.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set your API Key:**
    This project uses an environment variable to handle your API key securely. Before running the script, set the `GROQ_API_KEY` environment variable.

    ```bash
    export GROQ_API_KEY="your_api_key_here"
    ```

## Replicating the Experiments

The main script for running experiments is `scripts/run_experiment.py`. You can run different types of experiments using command-line arguments.

### Generating a Full Report Table

To run all conditions (baseline, adversarial, supportive, etc.) and generate a summary table in a Markdown file, use the `--full_report` flag.

```bash
python scripts/run_experiment.py data/toefl4.jsonl results/toefl4_full --full_report --mode agentic --runs 3
```
This command will run the agentic model three times on all conditions and save the aggregated results table to `runs/toefl4_full_report.md`.

### Running an Ablation Study

To run the ablation study, which requires pre-analyzed input files:

```bash
python scripts/run_experiment.py data/toefl4_analyzed.jsonl results/toefl4_ablation --run_ablation_study --resynthesize_only --runs 3
```
This will generate the ablation results table in `runs/toefl4_ablation_report.md`.


## Repository Structure

```
.
├── data/
│   ├── toefl4.jsonl
│   └── ... (other datasets)
├── scripts/
│   ├── run_experiment.py   # Main script to run all experiments
│   └── redact_data.py      # Preprocessing script to redact data
├── src/
│   ├── __init__.py
│   ├── constants.py        # Prompts, languages, and other constants
│   ├── evaluation.py       # Calculates accuracy and F1-score
│   ├── experiment.py       # Core experiment logic
│   ├── llm_api.py          # Handles API calls to the language model
│   ├── reporting.py        # Generates the final results tables
│   └── utils.py            # Utility functions
├── requirements.txt        # Python dependencies
└── README.md
```