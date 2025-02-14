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

def select_best_image(species_dir, image_files, feature_type, used_images):
    """Select the most appropriate image based on feature type."""
    # Keywords to look for in filenames or analyze images
    keywords = {
        'tree_form': ['full', 'tree', 'shape', 'silhouette', 'crown'],
        'bark': ['bark', 'trunk', 'stem', 'texture'],
        'leaf': ['leaf', 'needle', 'foliage', 'branch']
    }
    
    # Try to find an image with matching keywords that hasn't been used
    for img in image_files:
        if img in used_images:
            continue
        img_lower = img.lower()
        for keyword in keywords.get(feature_type, []):
            if keyword in img_lower:
                used_images.add(img)
                return img
    
    # If no keyword match, use heuristics based on image properties
    for img in image_files:
        if img in used_images:
            continue
        try:
            with Image.open(os.path.join(species_dir, img)) as im:
                width, height = im.size
                if feature_type == 'tree_form' and width < height:
                    used_images.add(img)
                    return img
                elif feature_type == 'bark' and abs(width - height) < 50:
                    used_images.add(img)
                    return img
                elif feature_type == 'leaf' and width > height:
                    used_images.add(img)
                    return img
        except Exception:
            continue
    
    # Fallback to first unused image
    for img in image_files:
        if img not in used_images:
            used_images.add(img)
            return img
    
    return None

def create_species_comparison(dataset_dir, species_groups, output_dir):
    """Create comparison visualizations for similar species with dataset quality notes."""
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

    # Selected representative images and dataset quality notes
    selected_images = {
        'needle_trees': {
            'jelka': {
                'image': 'jelka_032.jpg',
                'description': 'Silver Fir - Pyramidal shape with horizontal branches',
                'dataset_note': 'Limited high-quality images available. Many photos lack clear needle detail or full tree perspective.'
            },
            'smreka': {
                'image': 'smreka_028.jpg',
                'description': 'Norway Spruce - Conical shape with drooping branches',
                'dataset_note': 'Variable image quality. Seasonal variations and mixed forest backgrounds make identification challenging.'
            },
            'bor': {
                'image': 'bor_001.jpg',
                'description': 'Scots Pine - Broad, irregular crown with distinctive bark',
                'dataset_note': 'Inconsistent lighting conditions. Many images show only partial views or are taken from suboptimal angles.'
            }
        },
        'broad_trees': {
            'gaber': {
                'image': 'gaber_028.jpg',
                'description': 'Common Hornbeam - Dense crown with upright branches',
                'dataset_note': 'Small sample size affects representation. Many images lack clear distinguishing features.'
            },
            'bukev': {
                'image': 'bukev_030.jpg',
                'description': 'Common Beech - Wide crown with smooth, silver-gray bark',
                'dataset_note': 'Seasonal variations in foliage. Limited variety in viewing angles and distances.'
            },
            'javor': {
                'image': 'javor_047.jpg',
                'description': 'Sycamore Maple - Broad crown with distinctive leaves',
                'dataset_note': 'Mixed quality in training data. Background clutter often obscures key features.'
            }
        }
    }

    for group_name, species_list in species_groups.items():
        # Create figure with improved spacing
        fig = plt.figure(figsize=(15, 5*len(species_list)))
        fig.suptitle(f'Comparison of {group_name.replace("_", " ").title()}\nDataset Quality Analysis', 
                    fontsize=16, y=0.98)
        
        # Create grid with better spacing
        gs = fig.add_gridspec(len(species_list), 2, width_ratios=[1, 1.5],
                            hspace=0.4, wspace=0.3)
        
        for i, species in enumerate(species_list):
            species_info = selected_images[group_name][species]
            species_dir = os.path.join(dataset_dir, 'train', species)
            
            # Image subplot
            ax_img = fig.add_subplot(gs[i, 0])
            img_path = os.path.join(species_dir, species_info['image'])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Center crop to square if needed
                width, height = img.size
                if width != height:
                    size = min(width, height)
                    left = (width - size) // 2
                    top = (height - size) // 4  # Crop from upper portion for trees
                    img = img.crop((left, top, left + size, top + size))
                ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title(species_names.get(species, species), fontsize=12, pad=10)
            
            # Text subplot
            ax_text = fig.add_subplot(gs[i, 1])
            description = species_info['description']
            dataset_note = species_info['dataset_note']
            text = f"{description}\n\nDataset Limitations:\n{dataset_note}"
            ax_text.text(0.05, 0.5, text, 
                        ha='left', va='center',
                        fontsize=10, wrap=True,
                        bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='lightgray', boxstyle='round'))
            ax_text.axis('off')
        
        # Adjust layout
        plt.subplots_adjust(top=0.95, bottom=0.05, 
                          left=0.05, right=0.95,
                          hspace=0.4)
        
        output_filename = f"{group_name}_comparison.png"
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