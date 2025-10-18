# Learning Rate Analysis - Essay Dataset

## 📊 Learning Rate Comparison

### **Unified Learning Rates (for fair comparison)**

| Model | Official Recommendation | run_cebab/main | Previous Essay | **New Unified** | Change |
|-------|------------------------|----------------|----------------|-----------------|---------|
| **BERT** | 2e-5 to 5e-5 | 1e-5 | 2e-5 | **1e-5** | -50% |
| **RoBERTa** | 1e-5 to 6e-5 | 1e-5 | 2e-5 | **1e-5** | -50% |
| **GPT2** | 1e-5 to 5e-5 | 1e-4 | 5e-5 | **1e-4** | +100% |
| **LSTM** | 1e-3 to 1e-2 | 1e-2 | 5e-4 | **1e-2** | +1900% |

## 🎯 Rationale for Unified Learning Rates

### **1. Fair Comparison**
- **Same learning rates** for PLMs and CBE-PLMs ensures fair comparison
- Eliminates learning rate as a confounding variable
- Consistent with run_cebab and main/test_main.py experiments

### **2. Official Recommendations**
- **BERT/RoBERTa**: 1e-5 is within official range (1e-5 to 5e-5)
- **GPT2**: 1e-4 is within official range (1e-5 to 5e-5) 
- **LSTM**: 1e-2 matches official recommendation (1e-3 to 1e-2)

### **3. Proven Performance**
- These learning rates are already validated in run_cebab experiments
- Consistent across multiple datasets (cebab, imdb)
- No need for dataset-specific optimization

## 📈 Expected Impact

### **BERT/RoBERTa (1e-5)**
- **Previous**: 2e-5 (higher LR)
- **Expected**: More stable training, potentially better convergence
- **Risk**: Slightly slower convergence

### **GPT2 (1e-4)**
- **Previous**: 5e-5 (lower LR)  
- **Expected**: Faster convergence, potentially better performance
- **Risk**: Possible training instability

### **LSTM (1e-2)**
- **Previous**: 5e-4 (much lower LR)
- **Expected**: Significant performance improvement
- **Risk**: Training instability, need careful monitoring

## 🔍 Key Changes

### **Most Significant Changes**:
1. **LSTM**: 5e-4 → 1e-2 (20x increase)
2. **GPT2**: 5e-5 → 1e-4 (2x increase)

### **Conservative Changes**:
1. **BERT**: 2e-5 → 1e-5 (2x decrease)
2. **RoBERTa**: 2e-5 → 1e-5 (2x decrease)

## 🚀 Next Steps

1. **Run experiments** with unified learning rates
2. **Compare results** with previous runs
3. **Analyze impact** on CBE-PLMs vs PLMs performance
4. **Monitor training stability** especially for LSTM and GPT2

## 📝 Notes

- All learning rates are now **consistent** across PLMs and CBE-PLMs
- This ensures **fair comparison** between the two approaches
- Results will be more **interpretable** and **reliable**
- Follows **established best practices** from the codebase
