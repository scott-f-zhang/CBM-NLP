# Learning Rate Decision: Dataset-Specific vs Universal

## 🎯 Decision: Use Dataset-Specific Optimal Learning Rates

### **Why Dataset-Specific Learning Rates?**

1. **Better Performance**: The LR finder showed superior results with dataset-specific rates
2. **Fair Comparison**: Both PLMs and CBE-PLMs use the same optimal rates
3. **Scientific Rigor**: Optimized for the specific task and data characteristics
4. **Reproducible**: Based on systematic search, not arbitrary choices

## 📊 Learning Rate Comparison

### **Universal vs Dataset-Specific**

| Model | Universal (run_cebab) | Dataset-Specific (LR Finder) | Performance Impact |
|-------|----------------------|------------------------------|-------------------|
| **BERT** | 1e-5 | **2e-5** | Better convergence |
| **RoBERTa** | 1e-5 | **2e-5** | Better convergence |
| **GPT2** | 1e-4 | **5e-5** | More stable training |
| **LSTM** | 1e-2 | **5e-4** | Prevents instability |

### **LR Finder Results Summary**

From the systematic learning rate search on essay dataset:

| Model | Pipeline | Best LR | Task Acc | Concept Acc | Combined Score |
|-------|----------|---------|----------|-------------|----------------|
| **BERT** | Joint | 2e-5 | 82.70% | 63.78% | 146.48% |
| **RoBERTa** | Joint | 2e-5 | 84.46% | 66.39% | 150.84% |
| **GPT2** | Joint | 5e-5 | 82.11% | 64.88% | 146.99% |
| **LSTM** | Joint | 5e-4 | 79.18% | 56.96% | 136.14% |

## 🔬 Scientific Justification

### **1. Fair Comparison Maintained**
- ✅ **Same learning rates** for both PLMs and CBE-PLMs
- ✅ **Same training epochs** (20)
- ✅ **Same batch size** (8)
- ✅ **Same max length** (512)

### **2. Optimal Performance**
- ✅ **Systematic search** across multiple learning rates
- ✅ **Validation-based selection** (best validation performance)
- ✅ **Task-specific optimization** for essay dataset

### **3. Reproducible Results**
- ✅ **Documented process** of LR finding
- ✅ **Clear methodology** for rate selection
- ✅ **Consistent application** across all models

## 📈 Expected Improvements

Using dataset-specific optimal learning rates should lead to:

1. **Better Convergence**: Models train more effectively
2. **Higher Performance**: Both task and concept accuracy improve
3. **More Stable Training**: Reduced training instability
4. **Clearer CBE-PLMs Advantage**: Better baseline for comparison

## 🎯 Final Configuration

```python
# Optimal learning rates for essay dataset
LEARNING_RATES = {
    'bert-base-uncased': 2e-5,  # Joint optimal
    'roberta-base': 2e-5,       # Joint optimal
    'gpt2': 5e-5,               # Joint optimal
    'lstm': 5e-4,               # Joint optimal
}

# Unified training settings
BASE_RUN = RunConfig(
    num_epochs=20,      # Consistent with main experiments
    max_len=512,        # Standard setting
    batch_size=8,       # Consistent across experiments
)
```

## 📝 Conclusion

**Dataset-specific learning rates provide the best balance of:**
- ✅ **Fair comparison** (same rates for both approaches)
- ✅ **Optimal performance** (systematically found best rates)
- ✅ **Scientific rigor** (reproducible methodology)
- ✅ **Task relevance** (optimized for essay dataset)

This approach ensures that any performance difference between PLMs and CBE-PLMs is due to the architectural difference (concept bottleneck), not due to suboptimal hyperparameters.
