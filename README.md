# Native Language Identification through Agentic Decomposition

This repository contains the code and instructions to replicate the experiments from the paper Robust Native Language Identification through Agentic Decomposition at EMLNP 2025.

## Datasets

This research utilizes two main datasets: TOEFL11 and the Write & Improve 2024 corpus. A `data` directory should be created in the root of this project to store them.

### TOEFL11 Dataset

The TOEFL11 dataset is a corpus of essays from learners of English as a foreign language.

*   **Access:** To access this dataset, you will need to go through the official ETS request process. You can find more information here: [https://catalog.ldc.upenn.edu/LDC2014T06](https://catalog.ldc.upenn.edu/LDC2014T06)
*   **Setup:** Once obtained, place the dataset files into a `data/toefl11/` directory.

### Write & Improve 2024

The Write & Improve (W&I) corpus contains texts written by learners of English from across the world, which have been annotated.

*   **Access:** The W&I and LOCNESS corpora are available through Cambridge University. More information can be found at: [https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024](https://englishlanguageitutoring.com/datasets/write-and-improve-corpus-2024)
*   **Setup:** Once you have the data, create a `data/write_and_improve/` directory and place the files there.

## Setup

1.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Replicating the Experiments

2. The `analyze.py` script contains the necessary code to replicate the experiments from the paper.

## Repository Structure

    .
    ├── data/
    │   ├── toefl11/
    │   └── write_and_improve/
    ├── analyze.py
    ├── requirements.txt
    └── README.md
