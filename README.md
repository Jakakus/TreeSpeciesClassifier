# Tree Species Classification using Deep Learning

## Project Overview
This project implements a deep learning-based classification system for identifying tree species from images. Using EfficientNetB0 architecture and advanced training techniques, the model can classify 10 different tree species common in Slovenia.

## Key Features
- EfficientNetB0-based classification model
- Two-phase training strategy with transfer learning
- Advanced data augmentation pipeline
- Mixup training (α=0.3)
- Learning rate scheduling and early stopping
- Comprehensive performance monitoring

## Model Performance
- Overall Accuracy: 38.8% (significantly above random chance of 10%)
- Macro-average F1-score: 0.375
- Top-2 Accuracy: ~60%

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

#### Phase 2 (Fine-tuning)
- Unfrozen top layers
- 15 epochs
- Early stopping (patience=20)
- Fixed learning rate

### Data Augmentation
- Random rotation
- Random zoom
- Random flip
- Random brightness
- Mixup augmentation (α=0.3)

## Results Interpretation
### Strong Performance
- `kostanj`: Best performing class (F1=0.667)
- `hrast`: Strong secondary performer (F1=0.571)

### Areas for Improvement
- `jelka`: High precision but low recall
- `gaber`: Balanced but low metrics

## Project Structure
```
├── train_trees.py         # Main training script
├── download_trees.py      # Dataset preparation
├── trees_dataset/         # Image dataset
│   └── all_images/       # Organized by species
├── training_results/      # Training logs and visualizations
└── requirements.txt       # Dependencies
```

## Installation
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

## Usage
1. Prepare the dataset:
   ```bash
   python download_trees.py
   ```
2. Train the model:
   ```bash
   python train_trees.py
   ```

## Future Improvements
- Implement attention mechanisms
- Add hierarchical classification
- Enhance data augmentation for struggling classes
- Explore ensemble methods

## License
[MIT License](LICENSE)

## Author
[Jaka Kušar](https://github.com/Jakakus) 