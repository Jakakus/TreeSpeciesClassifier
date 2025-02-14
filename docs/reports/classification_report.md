# Classification Report

```
              precision    recall  f1-score   support

         bor      0.333     0.250     0.286         8
       bukev      0.333     0.250     0.286         8
       gaber      0.286     0.250     0.267         8
       hrast      0.667     0.500     0.571         8
       javor      0.429     0.375     0.400         8
       jelka      0.500     0.125     0.200         8
     kostanj      0.600     0.750     0.667         8
        lipa      0.312     0.625     0.417         8
     macesen      0.400     0.250     0.308         8
      smreka      0.267     0.500     0.348         8

    accuracy                          0.388        80
   macro avg      0.413     0.388     0.375        80
weighted avg      0.413     0.388     0.375        80

```

## Metric Explanations

### Precision
- Measures how many of the predicted instances for each class were correct
- Higher precision means fewer false positives
- Important when the cost of false positives is high

### Recall
- Measures how many of the actual instances of each class were correctly identified
- Higher recall means fewer false negatives
- Important when the cost of false negatives is high

### F1-score
- Harmonic mean of precision and recall
- Provides a single score that balances both metrics
- Useful when you need a balanced measure of performance

### Support
- Number of samples of each class in the test set
- Helps interpret the significance of the metrics
- Larger support means more reliable metrics
