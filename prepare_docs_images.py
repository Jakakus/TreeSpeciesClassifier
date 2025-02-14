"""
Documentation Image Preparation Script
-----------------------------------
This script prepares and organizes images and reports for GitHub documentation:
1. Creates sample image grids for each species
2. Prepares comparison visualizations
3. Organizes training results, logs, and reports
4. Formats classification reports for better readability

Author: Jaka Kus
Date: February 2024
License: MIT
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import pandas as pd

def create_directory_structure():
    """Create the documentation image directory structure."""
    dirs = [
        'docs/images/samples',
        'docs/images/analysis',
        'docs/images/results',
        'docs/reports'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def create_species_grid(dataset_dir, output_path, samples_per_species=1):
    """Create a grid of sample images for each tree species."""
    species = sorted(os.listdir(os.path.join(dataset_dir, 'train')))
    n_species = len(species)
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (n_species + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    fig.suptitle('Sample Images for Each Tree Species', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Species name mapping
    species_names = {
        'bor': 'Bor (Scots Pine)',
        'bukev': 'Bukev (Common Beech)',
        'gaber': 'Gaber (Common Hornbeam)',
        'hrast': 'Hrast (Sessile Oak)',
        'javor': 'Javor (Sycamore Maple)',
        'jelka': 'Jelka (Silver Fir)',
        'kostanj': 'Kostanj (Sweet Chestnut)',
        'lipa': 'Lipa (Large-leaved Lime)',
        'macesen': 'Macesen (European Larch)',
        'smreka': 'Smreka (Norway Spruce)'
    }
    
    for i, species_name in enumerate(species):
        if i < len(axes_flat):
            species_dir = os.path.join(dataset_dir, 'train', species_name)
            image_files = sorted(os.listdir(species_dir))[:samples_per_species]
            
            if image_files:
                img_path = os.path.join(species_dir, image_files[0])
                img = Image.open(img_path)
                axes_flat[i].imshow(img)
                axes_flat[i].set_title(species_names.get(species_name, species_name), 
                                     fontsize=10, pad=5)
            axes_flat[i].axis('off')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_species_comparison(dataset_dir, species_groups, output_dir):
    """Create comparison visualizations for similar species."""
    # Species name mapping for better labels
    species_names = {
        'bor': 'Bor (Scots Pine)',
        'bukev': 'Bukev (Common Beech)',
        'gaber': 'Gaber (Common Hornbeam)',
        'hrast': 'Hrast (Sessile Oak)',
        'javor': 'Javor (Sycamore Maple)',
        'jelka': 'Jelka (Silver Fir)',
        'kostanj': 'Kostanj (Sweet Chestnut)',
        'lipa': 'Lipa (Large-leaved Lime)',
        'macesen': 'Macesen (European Larch)',
        'smreka': 'Smreka (Norway Spruce)'
    }

    feature_descriptions = {
        'needle_trees': {
            'title': 'Comparison of Needle-Leaved Trees',
            'features': ['Close-up Features', 'Full Tree View', 'Growth Pattern'],
            'descriptions': {
                'jelka': [
                    'Flat needles arranged in rows',
                    'Tall, straight trunk',
                    'Regular, symmetrical branching'
                ],
                'smreka': [
                    'Dense needle clusters',
                    'Conical shape with regular branches',
                    'Uniform growth pattern'
                ],
                'bor': [
                    'Long needles with cones',
                    'Mature tree with spreading crown',
                    'Young tree in cultivation'
                ]
            }
        },
        'broad_trees': {
            'title': 'Comparison of Broad-Leaved Trees',
            'features': ['Leaf Detail', 'Tree Form', 'Growth Characteristics'],
            'descriptions': {
                'gaber': [
                    'Distinctive hanging seed clusters',
                    'Dense leaf arrangement',
                    'Branch and leaf pattern'
                ],
                'bukev': [
                    'Typical leaf arrangement',
                    'Mature tree form',
                    'Characteristic branching'
                ],
                'javor': [
                    'Full tree silhouette',
                    'Distinctive maple leaves',
                    'Mature tree shape'
                ]
            }
        }
    }

    for group_name, species_list in species_groups.items():
        group_info = feature_descriptions['needle_trees' if group_name == 'needle_trees' else 'broad_trees']
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=(15, 7*len(species_list)))
        
        # Add title with more space at top
        fig.suptitle(group_info['title'], fontsize=16, y=0.98)
        
        # Create grid with proper spacing for headers and descriptions
        height_ratios = []
        for _ in range(len(species_list)):
            height_ratios.extend([1, 0.3])  # Image row and description row for each species
            
        gs = fig.add_gridspec(len(species_list)*2, 3, height_ratios=height_ratios)
        
        # Add column headers at the top with more space
        for j, feature in enumerate(group_info['features']):
            fig.text(0.25 + j*0.25, 0.95, feature, 
                    ha='center', va='bottom', 
                    fontsize=14, weight='bold')
        
        for i, species in enumerate(species_list):
            # Add species name on the left with proper alignment
            fig.text(0.02, 0.85 - (i*0.30), species_names.get(species, species),
                    fontsize=12, weight='bold',
                    ha='left', va='center')
            
            species_dir = os.path.join(dataset_dir, 'train', species)
            image_files = sorted(os.listdir(species_dir))[:3]
            
            for j, img_file in enumerate(image_files):
                # Create subplot for image
                ax_img = fig.add_subplot(gs[i*2, j])
                img_path = os.path.join(species_dir, img_file)
                img = Image.open(img_path)
                ax_img.imshow(img)
                ax_img.axis('off')
                
                # Add description below the image
                ax_desc = fig.add_subplot(gs[i*2+1, j])
                ax_desc.text(0.5, 0.7, group_info['descriptions'][species][j],
                           ha='center', va='center',
                           fontsize=10, wrap=True)
                ax_desc.axis('off')
        
        # Adjust layout with more space between rows
        plt.subplots_adjust(top=0.92, bottom=0.02, 
                          left=0.15, right=0.95,
                          hspace=0.2, wspace=0.1)
        
        output_filename = 'needle_trees_comparison.png' if group_name == 'needle_trees' else 'broad_trees_comparison.png'
        plt.savefig(os.path.join(output_dir, output_filename),
                   dpi=300, bbox_inches='tight')
        plt.close()

def format_classification_report(input_path, output_path):
    """Format classification report for better readability in markdown."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            report = f.read()
        
        # Add markdown formatting
        formatted_report = "# Classification Report\n\n```\n" + report + "\n```\n\n"
        
        # Add explanations
        formatted_report += """## Metric Explanations

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
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
    except Exception as e:
        print(f"Warning: Could not format classification report: {str(e)}")
        print("Continuing with other documentation tasks...")

def analyze_training_logs(phase1_log, phase2_log, output_path):
    """Analyze and summarize training logs."""
    try:
        # Read logs
        df1 = pd.read_csv(phase1_log)
        df2 = pd.read_csv(phase2_log)
        
        # Create summary
        summary = """# Training Process Analysis

## Phase 1 (Feature Extraction)
- **Duration**: {} epochs
- **Best Validation Accuracy**: {:.2%}
- **Final Training Loss**: {:.3f}
- **Best Top-2 Accuracy**: {:.2%}

## Phase 2 (Fine-tuning)
- **Duration**: {} epochs
- **Best Validation Accuracy**: {:.2%}
- **Final Training Loss**: {:.3f}
- **Best Top-2 Accuracy**: {:.2%}

## Key Observations
1. Phase 1 showed {} improvement in validation accuracy
2. Phase 2 achieved {} stability in metrics
3. Early stopping triggered after {} epochs without improvement
4. Performance was best during {}

## Training Metrics
### Phase 1
- Average validation accuracy: {:.2%}
- Accuracy improvement: {:.2%} to {:.2%}
- Loss reduction: {:.2f} to {:.2f}

### Phase 2
- Average validation accuracy: {:.2%}
- Accuracy improvement: {:.2%} to {:.2%}
- Loss reduction: {:.2f} to {:.2f}
""".format(
            len(df1),
            df1['val_accuracy'].max(),
            df1['loss'].iloc[-1],
            df1['val_top_2_accuracy'].max(),
            
            len(df2),
            df2['val_accuracy'].max(),
            df2['loss'].iloc[-1],
            df2['val_top_2_accuracy'].max(),
            
            'steady' if df1['val_accuracy'].diff().mean() > 0 else 'variable',
            'consistent' if df2['val_accuracy'].std() < 0.05 else 'variable',
            max(len(df1), len(df2)),
            'Phase 1' if df1['val_accuracy'].max() > df2['val_accuracy'].max() else 'Phase 2',
            
            df1['val_accuracy'].mean(),
            df1['val_accuracy'].iloc[0], df1['val_accuracy'].iloc[-1],
            df1['loss'].iloc[0], df1['loss'].iloc[-1],
            
            df2['val_accuracy'].mean(),
            df2['val_accuracy'].iloc[0], df2['val_accuracy'].iloc[-1],
            df2['loss'].iloc[0], df2['loss'].iloc[-1]
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
    except Exception as e:
        print(f"Warning: Could not analyze training logs: {str(e)}")
        print("Continuing with other documentation tasks...")

def copy_training_results(results_dir, output_dir):
    """Copy and organize training result visualizations and reports."""
    # Copy images
    files_to_copy = {
        'combined_training_history.png': 'training_history.png',
        'confusion_matrix.png': 'confusion_matrix.png'
    }
    
    for src, dst in files_to_copy.items():
        src_path = os.path.join(results_dir, src)
        dst_path = os.path.join(output_dir, dst)
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Source file not found: {src_path}")
        except Exception as e:
            print(f"Warning: Could not copy {src}: {str(e)}")
    
    # Format classification report
    try:
        report_path = os.path.join(results_dir, 'classification_report.txt')
        if os.path.exists(report_path):
            format_classification_report(
                report_path,
                'docs/reports/classification_report.md'
            )
        else:
            print(f"Warning: Classification report not found: {report_path}")
    except Exception as e:
        print(f"Warning: Could not process classification report: {str(e)}")
    
    # Analyze training logs
    try:
        phase1_log = os.path.join(results_dir, 'phase1_training_log.csv')
        phase2_log = os.path.join(results_dir, 'phase2_training_log.csv')
        
        if os.path.exists(phase1_log) and os.path.exists(phase2_log):
            analyze_training_logs(
                phase1_log,
                phase2_log,
                'docs/reports/training_analysis.md'
            )
        else:
            print("Warning: Training logs not found")
    except Exception as e:
        print(f"Warning: Could not analyze training logs: {str(e)}")

    print("\nDocumentation preparation completed!")
    print("Note: Some warnings may have occurred. Check the messages above.")

def create_species_examples(dataset_dir, species_name, output_dir, n_examples=3):
    """Create a visualization of example images for a specific species."""
    species_dir = os.path.join(dataset_dir, 'train', species_name)
    if not os.path.exists(species_dir):
        print(f"Warning: Directory not found for species {species_name}")
        return
    
    image_files = sorted(os.listdir(species_dir))[:n_examples]
    if len(image_files) < n_examples:
        print(f"Warning: Not enough images found for {species_name}")
        return
    
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 5))
    fig.suptitle(f'Example Images: {species_name}', fontsize=16)
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(species_dir, img_file)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Example {chr(65+i)}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{species_name.lower()}_examples.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directory structure
    create_directory_structure()
    
    # Define paths
    dataset_dir = 'trees_dataset'
    results_dir = 'training_results'
    
    # Create species grid
    create_species_grid(
        dataset_dir,
        'docs/images/samples/tree_species_grid.png'
    )
    
    # Create individual species examples
    for species in ['kostanj', 'hrast']:
        create_species_examples(
            dataset_dir,
            species,
            'docs/images/analysis'
        )
    
    # Create species comparisons with better names and organization
    species_groups = {
        'needle_trees': ['jelka', 'smreka', 'bor'],
        'broad_trees': ['gaber', 'bukev', 'javor']
    }
    create_species_comparison(
        dataset_dir,
        species_groups,
        'docs/images/analysis'
    )
    
    # Copy and organize training results
    copy_training_results(
        results_dir,
        'docs/images/results'
    )

if __name__ == '__main__':
    main() 