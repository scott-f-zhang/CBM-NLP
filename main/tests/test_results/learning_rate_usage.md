# Learning Rate Configuration Usage

## 🎯 Overview

The `get_learning_rate` function now supports switching between two learning rate configurations:

1. **`dataset_optimal`**: Optimal learning rates found by LR finder for essay dataset
2. **`universal`**: Universal learning rates consistent with run_cebab/main

## 🔧 Configuration

### **Method 1: Modify LR_TYPE in test_essay.py**

```python
# In main/tests/test_essay.py
LR_TYPE = "dataset_optimal"  # or "universal"
```

### **Method 2: Use the comparison script**

```bash
# Run both configurations and compare results
python run_essay_comparison.py
```

## 📊 Learning Rate Values

### **Dataset-Optimal Learning Rates**
| Model | Learning Rate | Source |
|-------|---------------|--------|
| **BERT** | 2e-5 | LR finder (Joint optimal) |
| **RoBERTa** | 2e-5 | LR finder (Joint optimal) |
| **GPT2** | 5e-5 | LR finder (Joint optimal) |
| **LSTM** | 5e-4 | LR finder (Joint optimal) |

### **Universal Learning Rates**
| Model | Learning Rate | Source |
|-------|---------------|--------|
| **BERT** | 1e-5 | run_cebab/main consistency |
| **RoBERTa** | 1e-5 | run_cebab/main consistency |
| **GPT2** | 1e-4 | run_cebab/main consistency |
| **LSTM** | 1e-2 | run_cebab/main consistency |

## 🚀 Usage Examples

### **Run with Dataset-Optimal Learning Rates**
```bash
# Set LR_TYPE = "dataset_optimal" in test_essay.py
python main/run_essay.py
# Results saved to: result_essay_dataset_optimal.csv
```

### **Run with Universal Learning Rates**
```bash
# Set LR_TYPE = "universal" in test_essay.py
python main/run_essay.py
# Results saved to: result_essay_universal.csv
```

### **Run Both and Compare**
```bash
python run_essay_comparison.py
# Runs both configurations and compares results
```

## 📈 Expected Results

### **Dataset-Optimal Learning Rates**
- ✅ **Better performance** on essay dataset
- ✅ **Optimized for specific task**
- ✅ **Higher CBE-PLMs advantage**

### **Universal Learning Rates**
- ✅ **Consistent with other experiments**
- ✅ **Fair comparison baseline**
- ✅ **Reproducible across datasets**

## 🔬 Scientific Rigor

Both configurations maintain:
- ✅ **Same learning rates** for PLMs and CBE-PLMs
- ✅ **Same training epochs** (20)
- ✅ **Same batch size** (8)
- ✅ **Same max length** (512)
- ✅ **Fair comparison** between approaches

## 📝 Recommendations

1. **For Essay Dataset**: Use `dataset_optimal` for best performance
2. **For Cross-Dataset Comparison**: Use `universal` for consistency
3. **For Research**: Run both and compare results
4. **For Reproducibility**: Document which LR type was used

## 🎯 Default Setting

The default setting is `dataset_optimal` as it provides better performance on the essay dataset while maintaining fair comparison between PLMs and CBE-PLMs.
