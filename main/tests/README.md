# CBM-NLP Tests

This directory contains test scripts for the CBM-NLP project.

## Test Files

### Main Tests
- **`test_essay.py`**: Run essay dataset experiments (PLMs vs CBE-PLMs)
- **`test_early_stopping.py`**: Test Early Stopping functionality
- **`test_main.py`**: Lightweight test runner with reduced epochs

### Model-Specific Tests
- **`test_bert.py`**: BERT-only experiments
- **`test_roberta.py`**: RoBERTa-only experiments  
- **`test_gpt2.py`**: GPT2-only experiments
- **`test_lstm.py`**: LSTM-only experiments

### Data Tests
- **`test_dataclass_loading.py`**: Data loading functionality tests

## Running Tests

### From Project Root
```bash
# Run complete essay experiments (recommended - NO early stopping by default)
python main/run_essay.py

# Run with early stopping enabled
python main/run_essay.py --early-stopping

# Run with early stopping explicitly disabled
python main/run_essay.py --no-early-stopping

# Run essay experiments directly (with early stopping)
python main/tests/test_essay.py

# Run essay experiments without early stopping
python main/tests/test_essay_no_early_stopping.py

# Test early stopping
python main/tests/test_early_stopping.py

# Run lightweight tests
python main/tests/test_main.py

# Run model-specific tests
python main/tests/test_bert.py
python main/tests/test_roberta.py
python main/tests/test_gpt2.py
python main/tests/test_lstm.py
```

### From Main Directory
```bash
cd main

# Run complete essay experiments (recommended - NO early stopping by default)
python run_essay.py

# Run with early stopping enabled
python run_essay.py --early-stopping

# Run essay experiments directly (with early stopping)
python tests/test_essay.py

# Run essay experiments without early stopping
python tests/test_essay_no_early_stopping.py
```

### From Tests Directory
```bash
cd main/tests

# Run essay experiments (with early stopping)
python test_essay.py

# Run essay experiments without early stopping
python test_essay_no_early_stopping.py

# Test early stopping
python test_early_stopping.py

# Run lightweight tests
python test_main.py
```

## Environment Setup

Make sure to activate the conda environment before running tests:
```bash
conda activate cbm
```

## Test Results

All test results are automatically saved to the `test_results/` directory:

```
main/tests/test_results/
├── result_essay_early_stopping_dataset_optimal.csv      # Essay experiments WITH early stopping
├── result_essay_no_early_stopping_dataset_optimal.csv   # Essay experiments WITHOUT early stopping (recommended)
├── result_test.csv            # Lightweight test results
├── result_bert_test.csv       # BERT-specific test results
├── result_roberta_test.csv    # RoBERTa-specific test results
├── result_gpt2_test.csv       # GPT2-specific test results
├── result_lstm_test.csv       # LSTM-specific test results
├── lr_finder_results.csv      # Learning rate finder results
├── early_stopping_analysis.md # Analysis of early stopping impact
└── lr_analysis_summary.md     # Learning rate analysis summary
```

## Notes

- All test files have been updated with correct import paths
- All results are automatically saved to `main/tests/test_results/` directory
- **Default behavior**: NO early stopping for fair comparison (recommended)
- **Early stopping option**: Available via `--early-stopping` flag
- Early Stopping is implemented in both PLMs and CBE-PLMs pipelines
- Tests use the essay dataset with 7:2:1 train/dev/test split
- **Recommended**: Use `python main/run_essay.py` (no early stopping) for fair comparison
