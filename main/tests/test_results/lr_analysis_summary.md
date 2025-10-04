# Learning Rate Analysis Summary - Essay Dataset (7:2:1 Split)

## Results Overview

Based on the latest learning rate finder run on the essay dataset with 7:2:1 train/dev/test split:

| Model | Pipeline | Best LR | Task Acc | Task F1 | Concept Acc | Concept F1 | Combined Score |
|-------|----------|---------|----------|---------|-------------|------------|----------------|
| bert-base-uncased | standard | 5e-05 | 82.40% | 68.84% | 0.00% | 0.00% | 82.40% |
| bert-base-uncased | joint | 2e-05 | 82.70% | 66.26% | 63.78% | 64.84% | 146.48% |
| roberta-base | standard | 1e-05 | 83.58% | 69.20% | 0.00% | 0.00% | 83.58% |
| roberta-base | joint | 2e-05 | 84.46% | 70.17% | 66.39% | 69.11% | 150.84% |
| gpt2 | standard | 2e-04 | 82.70% | 60.35% | 0.00% | 0.00% | 82.70% |
| gpt2 | joint | 5e-05 | 82.11% | 62.70% | 64.88% | 68.32% | 146.99% |
| lstm | standard | 1e-04 | 79.18% | 44.19% | 0.00% | 0.00% | 79.18% |
| lstm | joint | 5e-04 | 79.18% | 44.19% | 56.96% | 53.77% | 136.14% |

## Key Observations

### 1. **Model Performance Ranking (by Combined Score)**
1. **RoBERTa Joint**: 150.84% (best overall)
2. **GPT2 Joint**: 146.99%
3. **BERT Joint**: 146.48%
4. **LSTM Joint**: 136.14%

### 2. **Learning Rate Patterns**
- **BERT/RoBERTa**: Similar optimal LRs (1e-5 to 5e-5)
- **GPT2**: Higher LR for standard (2e-4), lower for joint (5e-5)
- **LSTM**: Highest LRs overall (1e-4 to 5e-4)

### 3. **Pipeline Comparison**
- **Joint vs Standard**: Joint pipelines consistently achieve higher combined scores
- **Concept Learning**: Joint pipelines successfully learn concepts (60-66% accuracy)
- **Task Performance**: Standard pipelines sometimes achieve higher task accuracy

### 4. **Model-Specific Insights**
- **RoBERTa**: Best overall performance, good balance of task and concept learning
- **BERT**: Solid performance, slightly lower than RoBERTa
- **GPT2**: Good concept learning but lower task F1 scores
- **LSTM**: Lowest performance, struggles with both task and concept learning

## Recommendations

1. **Use RoBERTa Joint** for best overall performance
2. **Use BERT Joint** as a reliable alternative
3. **Avoid LSTM** for this dataset due to poor performance
4. **Consider GPT2** if concept learning is prioritized over task F1

## Next Steps

1. Run full experiments with these optimal learning rates
2. Compare results with previous data splits
3. Analyze concept learning patterns across models
4. Investigate why LSTM performs poorly on this dataset
