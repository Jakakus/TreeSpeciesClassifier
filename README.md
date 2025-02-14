# Tree Species Classification using Deep Learning

## Project Overview
This project implements a deep learning-based classification system for identifying tree species from images. Using EfficientNetB0 architecture and advanced training techniques, the model can classify 10 different tree species common in Slovenia.

## Visual Examples

### Sample Images per Species
![Tree Species Examples](images/samples/tree_species_grid.png)
*Sample images showing characteristic features of each tree species*

### Best Performing Classes

#### 1. Kostanj (Sweet Chestnut)
![Kostanj Examples](images/analysis/kostanj_examples.png)
*Distinctive features: (A) Spiral bark pattern (B) Serrated leaves (C) Fruit clusters*

#### 2. Hrast (Sessile Oak)
![Hrast Examples](images/analysis/hrast_examples.png)
*Distinctive features: (A) Rough bark texture (B) Lobed leaves (C) Characteristic branching*

### Challenging Classes

#### 1. Jelka vs Similar Species
![Conifer Comparison](images/analysis/conifer_comparison.png)
*Visual comparison showing similarity between Jelka, Smreka, and Bor*

#### 2. Gaber vs Similar Species
![Deciduous Comparison](images/analysis/deciduous_comparison.png)
*Visual comparison showing similarity between Gaber, Bukev, and Javor*

## Key Features
- EfficientNetB0-based classification model
- Two-phase training strategy with transfer learning
- Advanced data augmentation pipeline
- Mixup training (α=0.3)
- Learning rate scheduling and early stopping
- Comprehensive performance monitoring

## Model Performance

### Overall Metrics
- Overall Accuracy: 38.8% (significantly above random chance of 10%)
- Macro-average F1-score: 0.375
- Top-2 Accuracy: ~60%

### Training Results

#### Learning Curves
![Training History](images/results/combined_training_history.png)
*Training and validation metrics over both phases*

#### Confusion Matrix
![Confusion Matrix](images/results/confusion_matrix.png)
*Normalized confusion matrix showing inter-class confusion patterns*

### Class-specific Performance
| Species  | Precision | Recall | F1-Score |
|----------|-----------|--------|-----------|
| kostanj  | 0.600     | 0.750  | 0.667    |
| hrast    | 0.667     | 0.500  | 0.571    |
| jelka    | 0.500     | 0.125  | 0.200    |
| gaber    | 0.250     | 0.286  | 0.267    |

## Technical Implementation

### Model Architecture
- Base: EfficientNetB0
- Custom dense layers (384->192)
- Dropout layers for regularization
- Softmax output for 10 classes

### Training Strategy

#### Phase 1 (Feature Extraction)
- Frozen EfficientNetB0 layers
- 20 epochs
- Early stopping (patience=15)
- Learning rate reduction on plateau
- Results:
  - Initial accuracy: 14% → Final accuracy: 32%
  - Loss reduction: 11.3 → 9.2
  - Steady improvement, no overfitting

#### Phase 2 (Fine-tuning)
- Unfrozen top layers
- 15 epochs
- Early stopping (patience=20)
- Fixed learning rate
- Results:
  - Accuracy improvement: 33% → 36%
  - Loss stabilization around 9.1-9.2
  - Stable validation metrics

### Data Augmentation
- Random rotation
- Random zoom
- Random flip
- Random brightness
- Mixup augmentation (α=0.3)

## Results Analysis

### Strong Performance
1. **Kostanj (Sweet Chestnut)**
   - F1-score: 0.667
   - Precision: 0.600
   - Recall: 0.750
   - Success factors: Distinctive bark patterns and leaf structure

2. **Hrast (Sessile Oak)**
   - F1-score: 0.571
   - Precision: 0.667
   - Recall: 0.500
   - Success factors: Characteristic branching pattern and bark texture

### Areas for Improvement
1. **Jelka (Silver Fir)**
   - High precision but very low recall (0.125)
   - Challenge: Model is overly conservative
   - Cause: Visual similarity with other conifers

2. **Gaber (Common Hornbeam)**
   - Balanced but low precision and recall
   - Challenge: Frequently confused with similar deciduous trees
   - Cause: Shared characteristics with bukev and javor

## Project Structure
```
├── train_trees.py         # Main training script
├── download_trees.py      # Dataset preparation
├── trees_dataset/         # Image dataset
│   └── all_images/       # Organized by species
├── training_results/      # Training logs and visualizations
└── requirements.txt       # Dependencies
```

## Installation and Usage

### Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Prepare the dataset:
   ```bash
   python download_trees.py
   ```
2. Train the model:
   ```bash
   python train_trees.py
   ```

## Future Improvements

### Technical Enhancements
1. Implement attention mechanisms
2. Add hierarchical classification (e.g., separate conifer/deciduous first)
3. Enhance data augmentation for struggling classes
4. Explore ensemble methods

### Data Collection
1. Gather more examples of challenging species
2. Include seasonal variations
3. Add metadata (season, location, age)

### Deployment Considerations
1. Add confidence thresholds for predictions
2. Implement multi-view analysis
3. Consider mobile-optimized model versions

## Additional Resources

### Detailed Reports
- [Classification Report](reports/classification_report.md): Comprehensive metrics for each species
- [Training Analysis](reports/training_analysis.md): Detailed training process insights

### Training Logs
- [Phase 1 Training Log](training_results/phase1_training_log.csv): Feature extraction phase metrics
- [Phase 2 Training Log](training_results/phase2_training_log.csv): Fine-tuning phase metrics

### Source Code Documentation
- [train_trees.py](train_trees.py): Main training script
- [download_trees.py](download_trees.py): Dataset preparation script
- [prepare_docs_images.py](prepare_docs_images.py): Documentation preparation script

## License
[MIT License](LICENSE)

## Author
[Jaka Kus](https://github.com/Jakakus) 