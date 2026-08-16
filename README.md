# Switch-GShard-MoE Comparison

A comparative study of **Dense Feed-Forward Networks, Switch Mixture-of-Experts (MoE), and GShard-style MoE** for Natural Language Inference (NLI).

## Overview

This project compares three model architectures built on top of **DistilBERT**:

* **Dense Baseline** — conventional dense feed-forward network
* **Switch MoE** — top-1 expert routing
* **GShard MoE** — top-2 expert routing

All three models perform NLI classification using the same underlying DistilBERT encoder and compare model performance, parameter usage, inference latency, throughput, and expert utilization.

## Datasets

The training data combines three NLI datasets:

* **SNLI**
* **MultiNLI**
* **ANLI**

The validation evaluation includes:

* SNLI validation
* MultiNLI matched validation
* MultiNLI mismatched validation
* ANLI validation

Invalid labels are filtered before training and the training datasets are combined and shuffled.

## Model Architectures

### 1. Dense Baseline

The baseline uses a standard dense feed-forward network after the DistilBERT encoder.

```text
Input
  ↓
DistilBERT
  ↓
Dense FFN
  ↓
Classifier
  ↓
NLI Prediction
```

### 2. Switch MoE

The Switch model contains multiple experts and uses a router to select the **top-1 expert** for each token.

```text
Input
  ↓
DistilBERT
  ↓
Router
  ↓
Top-1 Expert
  ↓
Classifier
  ↓
NLI Prediction
```

The implementation uses **4 experts**.

### 3. GShard MoE

The GShard-style implementation uses **top-2 expert routing**, where each token is routed to its two highest-probability experts and their outputs are combined using normalized routing weights.

```text
Input
  ↓
DistilBERT
  ↓
Router
  ↓
Top-2 Experts
  ↓
Weighted Combination
  ↓
Classifier
  ↓
NLI Prediction
```

The implementation uses 4 experts and top-2 routing.

## Experimental Configuration

| Parameter                    |      Value |
| ---------------------------- | ---------: |
| Base Model                   | DistilBERT |
| Number of Experts            |          4 |
| Batch Size                   |         16 |
| GShard Batch Size            |          4 |
| GShard Gradient Accumulation |          4 |
| Epochs                       |         10 |
| Learning Rate                |       2e-5 |
| Maximum Sequence Length      |        128 |
| Optimizer                    |      AdamW |
| Auxiliary Loss Weight        |       0.05 |
| Gradient Clipping            |        1.0 |
| Early Stopping Patience      |          2 |

The configuration is defined directly in the experiment script.

## Evaluation Metrics

The project evaluates:

* Training loss
* Training accuracy
* SNLI validation accuracy
* MultiNLI matched accuracy
* MultiNLI mismatched accuracy
* ANLI validation accuracy
* Total parameters
* Active parameters
* Inference latency per batch
* Inference latency per sample
* Inference throughput
* Expert utilization

The script also generates comparison plots for accuracy, loss, parameter counts, latency, throughput, and expert usage.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/MICHELLE-HOOLGERI25/Switch-GShard-MoE-Comparison.git
cd Switch-GShard-MoE-Comparison
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the experiment

```bash
python Switch_GShard_MOE.py
```

The script automatically creates output directories for:

* Checkpoints
* Plots
* Logs
* CSV metrics
* Final summaries

The output directory is generated with a timestamp for each run.

## Output

The experiment produces:

```text
SG_outputs_safe_<timestamp>/
├── SG_checkpoints/
├── SG_plots/
├── SG_logs/
├── SG_csv/
└── SG_summary/
```

The final summary is saved in both **JSON** and **TXT** formats.

## Fairness Considerations

Inference benchmarking uses the same evaluation batch size for the three models.

However, training is **not strictly apples-to-apples** because GShard uses a smaller micro-batch size with gradient accumulation to reduce memory usage:

* Dense: batch size 16
* Switch: batch size 16
* GShard: micro-batch size 4 with gradient accumulation of 4

Therefore, training time should be interpreted carefully, while the inference benchmarking is designed to provide a fairer comparison.

## Project Structure

```text
Switch-GShard-MoE-Comparison/
│
├── Switch_GShard_MOE.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── results/
    └── summary/
        ├── final_summary_20260407_154812.json
        └── final_summary_20260407_154812.txt
```

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* Mixture-of-Experts
* Switch MoE
* GShard-style MoE
* Matplotlib
* Natural Language Inference

## Author

**Michelle Hoolgeri**

GitHub: [MICHELLE-HOOLGERI25](https://github.com/MICHELLE-HOOLGERI25)
