# Model Inference Examples

This directory contains example scripts demonstrating how to use trained models for inference on new texts.

## Files

### 1. `inference_single_text.py`
Simple inference example for new texts without labels.

**Features:**
- Load trained models from `saved_models/original/`
- Support both standard and joint modes
- Show predictions and probabilities
- Display concept predictions for joint mode
- Command-line interface

**Usage:**
```bash
# Basic usage with default text
python inference_single_text.py --model_name bert-base-uncased --mode standard

# Joint mode with custom text
python inference_single_text.py --model_name gpt2 --mode joint --text "This restaurant is amazing!"

# LSTM model
python inference_single_text.py --model_name lstm --mode standard
```

**Output Example (Standard Mode):**
```
Using device: cuda
Model: bert-base-uncased
Mode: standard
Text: This restaurant has amazing food and great service!
------------------------------------------------------------
Predicted Rating: 5 star(s)
Rating Probabilities:
  1 star: 0.0100 (1.00%)
  2 star: 0.0200 (2.00%)
  3 star: 0.0500 (5.00%)
  4 star: 0.1500 (15.00%)
  5 star: 0.7700 (77.00%)
```

**Output Example (Joint Mode):**
```
Using device: cuda
Model: bert-base-uncased
Mode: joint
Text: This restaurant has amazing food and great service!
------------------------------------------------------------
Task Prediction: 5 star(s)
Task Probabilities:
  1 star: 0.0100 (1.00%)
  2 star: 0.0200 (2.00%)
  3 star: 0.0500 (5.00%)
  4 star: 0.1500 (15.00%)
  5 star: 0.7700 (77.00%)

Concept Predictions:
  Food: Positive
    Probabilities: 0.050 (Neg) | 0.100 (Neu) | 0.850 (Pos)
  Ambiance: Neutral
    Probabilities: 0.150 (Neg) | 0.650 (Neu) | 0.200 (Pos)
  Service: Positive
    Probabilities: 0.030 (Neg) | 0.120 (Neu) | 0.850 (Pos)
  Noise: Neutral
    Probabilities: 0.200 (Neg) | 0.600 (Neu) | 0.200 (Pos)
```

### 2. `inference_with_metrics.py`
Complete inference example with metrics calculation.

**Features:**
- Load trained models from `saved_models/original/`
- Support both standard and joint modes
- Calculate accuracy and F1 scores
- Show per-sample results
- Support batch processing
- Load data from CSV files
- Built-in sample data for testing

**Usage:**
```bash
# Use sample data
python inference_with_metrics.py --model_name bert-base-uncased --mode standard

# Load from CSV file
python inference_with_metrics.py --model_name gpt2 --mode joint --csv_file test_data.csv

# Show detailed per-sample results
python inference_with_metrics.py --model_name roberta-base --mode standard --show_details

# Custom batch size
python inference_with_metrics.py --model_name lstm --mode joint --batch_size 16
```

**CSV File Format:**
```csv
text,label
"This restaurant is amazing!",4
"Food was terrible.",0
"Average experience.",2
```

**Output Example:**
```
Using device: cuda
Model: bert-base-uncased
Mode: standard
------------------------------------------------------------
Using sample data (use --csv_file to load your own data)
Loaded 8 samples

Results:
Accuracy: 0.8750 (87.50%)
Macro F1: 0.8667
Weighted F1: 0.8750

Detailed Results:
 1. This restaurant has amazing food and great service!...
    True: 5 star, Predicted: 5 star ✓
    Probabilities: [0.01 0.02 0.05 0.15 0.77]

 2. Food was terrible and service was awful....
    True: 1 star, Predicted: 1 star ✓
    Probabilities: [0.85 0.10 0.03 0.01 0.01]
```

## Required Model Files

The scripts expect trained models to be saved in the following structure:

```
saved_models/
└── original/
    ├── bert-base-uncased/
    │   ├── bert-base-uncased_model_standard.pth
    │   ├── bert-base-uncased_classifier_standard.pth
    │   ├── bert-base-uncased_joint.pth
    │   └── bert-base-uncased_ModelXtoCtoY_layer_joint.pth
    ├── gpt2/
    │   ├── gpt2_model_standard.pth
    │   ├── gpt2_classifier_standard.pth
    │   ├── gpt2_joint.pth
    │   └── gpt2_ModelXtoCtoY_layer_joint.pth
    └── ...
```

## Model Modes

### Standard Mode
- **Output**: Task prediction only (rating 1-5 stars)
- **Architecture**: BERT/GPT2/LSTM → Linear layers → Classification
- **Use case**: When you only need the final prediction

### Joint Mode
- **Output**: Task prediction + Concept predictions
- **Architecture**: BERT/GPT2/LSTM → ModelXtoCtoY → Task + Concepts
- **Use case**: When you need interpretable predictions (concept explanations)
- **Concepts**: Food, Ambiance, Service, Noise (each with Negative/Neutral/Positive)

## Supported Models

- `bert-base-uncased`: BERT model
- `gpt2`: GPT-2 model
- `lstm`: BiLSTM with attention
- `roberta-base`: RoBERTa model

## Command Line Arguments

### Common Arguments
- `--model_name`: Model to use (bert-base-uncased, gpt2, lstm, roberta-base)
- `--mode`: Model mode (standard, joint)
- `--device`: Device to use (auto, cpu, cuda)

### inference_single_text.py
- `--text`: Text to predict on (default: sample text)

### inference_with_metrics.py
- `--csv_file`: CSV file with text and label columns
- `--text_column`: Name of text column in CSV (default: 'text')
- `--label_column`: Name of label column in CSV (default: 'label')
- `--batch_size`: Batch size for inference (default: 8)
- `--show_details`: Show detailed per-sample results

## Error Handling

The scripts include comprehensive error handling:
- Check for model file existence
- Validate command line arguments
- Handle device availability
- Graceful error messages with suggestions

## Dependencies

- torch
- transformers
- scikit-learn
- pandas
- numpy

Make sure to install all dependencies before running the scripts:
```bash
pip install torch transformers scikit-learn pandas numpy
```
