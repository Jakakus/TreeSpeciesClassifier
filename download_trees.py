"""
Tree Species Dataset Creation Script
---------------------------------
This script downloads and processes images of tree species common in Slovenia.
It uses Bing Image Search to collect images and implements various processing
steps to create a clean, balanced dataset suitable for deep learning:

Features:
- Automatic image download from Bing
- Image preprocessing (resizing, format conversion)
- Dataset organization and cleanup
- Comprehensive error handling
- Quality verification checks

Author: Jaka Kus
Date: February 2024
License: MIT
"""

import os
import zipfile
from icrawler.builtin import BingImageCrawler
from PIL import Image
import shutil
import random

# Configuration constants
TARGET_SIZE = (224, 224)  # Standard size used in many CNN models
JPEG_QUALITY = 85        # Good balance between quality and file size

# Tree species dictionary with search terms in both Slovenian and Latin
TREE_SPECIES = {
    'smreka': ['navadna smreka drevo', 'picea abies tree', 'smreka v gozdu'],      # Norway Spruce
    'bukev': ['navadna bukev drevo', 'fagus sylvatica tree', 'bukev v gozdu'],     # Common Beech
    'jelka': ['bela jelka drevo', 'abies alba tree', 'jelka v gozdu'],             # Silver Fir
    'hrast': ['graden hrast drevo', 'quercus petraea tree', 'hrast v gozdu'],      # Sessile Oak
    'bor': ['rdeči bor drevo', 'pinus sylvestris tree', 'bor v gozdu'],           # Scots Pine
    'macesen': ['evropski macesen', 'larix decidua tree', 'macesen v gozdu'],      # European Larch
    'javor': ['gorski javor drevo', 'acer pseudoplatanus', 'javor v gozdu'],       # Sycamore Maple
    'kostanj': ['pravi kostanj drevo', 'castanea sativa', 'kostanj v gozdu'],      # Sweet Chestnut
    'lipa': ['lipa drevo slovenija', 'tilia platyphyllos', 'lipa v gozdu'],        # Large-leaved Lime
    'gaber': ['beli gaber drevo', 'carpinus betulus', 'gaber v gozdu']             # Common Hornbeam
}

def download_and_process_images(species_name, search_terms, output_dir, target_count=50):
    """Download and process images for a specific tree species.
    
    Args:
        species_name (str): Name of the tree species
        search_terms (list): List of search terms for image crawling
        output_dir (str): Base directory for saving processed images
        target_count (int): Number of images to collect per species (default: 50)
    
    Returns:
        bool: True if successfully collected target_count images, False otherwise
    """
    temp_dir = os.path.join(output_dir, f"temp_{species_name}")
    final_dir = os.path.join(output_dir, 'all_images', species_name)  # Changed to put in all_images directory
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    processed_count = 0
    
    # Try each search term until we get enough images
    for search_term in search_terms:
        if processed_count >= target_count:
            break
            
        print(f"Downloading images for '{search_term}'...")
        crawler = BingImageCrawler(storage={'root_dir': temp_dir})
        crawler.crawl(keyword=search_term, max_num=30, file_idx_offset=0)
        
        # Process downloaded images
        for filename in os.listdir(temp_dir):
            if processed_count >= target_count:
                break
                
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(temp_dir, filename)
            try:
                with Image.open(input_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Resize image
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # Save processed image
                    output_filename = f"{species_name}_{processed_count+1:03d}.jpg"
                    output_path = os.path.join(final_dir, output_filename)
                    img.save(output_path, "JPEG", quality=JPEG_QUALITY)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
            # Clean up original file
            try:
                os.remove(input_path)
            except:
                pass
    
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return processed_count == target_count

def create_dataset_splits(base_dir, train_split=0.7, val_split=0.15):
    """Create train/validation/test splits from the processed images.
    
    Args:
        base_dir (str): Base directory containing the images
        train_split (float): Proportion of data for training (default: 0.7)
        val_split (float): Proportion of data for validation (default: 0.15)
        
    Note: The remaining proportion (1 - train_split - val_split) will be used for testing.
    """
    splits = {
        'train': os.path.join(base_dir, 'train'),
        'validation': os.path.join(base_dir, 'validation'),
        'test': os.path.join(base_dir, 'test')
    }
    
    # Create split directories
    for split_dir in splits.values():
        os.makedirs(split_dir, exist_ok=True)
    
    # Process each species directory
    for species in os.listdir(base_dir):
        species_dir = os.path.join(base_dir, species)
        if not os.path.isdir(species_dir) or species in splits.keys():
            continue
            
        # Create species directories in each split
        for split_dir in splits.values():
            os.makedirs(os.path.join(split_dir, species), exist_ok=True)
        
        # Get all images and shuffle them
        images = [f for f in os.listdir(species_dir) if f.endswith('.jpg')]
        random.shuffle(images)
        
        # Calculate split sizes
        total = len(images)
        train_size = int(total * train_split)
        val_size = int(total * val_split)
        
        # Split images
        train_imgs = images[:train_size]
        val_imgs = images[train_size:train_size + val_size]
        test_imgs = images[train_size + val_size:]
        
        # Move images to respective splits
        for img in train_imgs:
            src = os.path.join(species_dir, img)
            dst = os.path.join(splits['train'], species, img)
            shutil.copy2(src, dst)
            
        for img in val_imgs:
            src = os.path.join(species_dir, img)
            dst = os.path.join(splits['validation'], species, img)
            shutil.copy2(src, dst)
            
        for img in test_imgs:
            src = os.path.join(species_dir, img)
            dst = os.path.join(splits['test'], species, img)
            shutil.copy2(src, dst)
        
        # Remove original directory after splitting
        shutil.rmtree(species_dir)

def create_zip_from_directory(input_dir, zip_filename):
    """Creates a zip archive containing all files in input_dir."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=input_dir)
                zf.write(filepath, arcname)
    print(f"Created zip archive: {zip_filename}")

def verify_dataset(base_dir):
    """Verify the dataset structure and image properties.
    
    Performs comprehensive checks on:
    - Directory structure
    - Image counts per split
    - Image dimensions
    - Image format and mode
    
    Args:
        base_dir (str): Base directory containing the dataset
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    print("\nVerifying dataset structure...")
    
    # Expected counts
    EXPECTED_COUNTS = {
        'train': 35,      # 70% of 50
        'validation': 8,  # 15% of 50
        'test': 7        # 15% of 50
    }
    
    # Check each split
    for split, expected_count in EXPECTED_COUNTS.items():
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"ERROR: Missing {split} directory")
            return False
            
        # Check each species directory
        for species in TREE_SPECIES.keys():
            species_dir = os.path.join(split_dir, species)
            if not os.path.exists(species_dir):
                print(f"ERROR: Missing {species} directory in {split}")
                return False
            
            # Count images
            images = [f for f in os.listdir(species_dir) if f.endswith('.jpg')]
            if len(images) != expected_count:
                print(f"ERROR: {split}/{species} has {len(images)} images, expected {expected_count}")
                return False
            
            # Verify image format
            for img_file in images:
                img_path = os.path.join(species_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        if img.size != TARGET_SIZE:
                            print(f"ERROR: {img_path} has wrong size {img.size}, expected {TARGET_SIZE}")
                            return False
                        if img.mode != "RGB":
                            print(f"ERROR: {img_path} has wrong mode {img.mode}, expected RGB")
                            return False
                except Exception as e:
                    print(f"ERROR: Cannot open {img_path}: {e}")
                    return False
    
    print("Dataset verification completed successfully!")
    print(f"Train images per class: {EXPECTED_COUNTS['train']}")
    print(f"Validation images per class: {EXPECTED_COUNTS['validation']}")
    print(f"Test images per class: {EXPECTED_COUNTS['test']}")
    return True

if __name__ == "__main__":
    base_dir = "./trees_dataset"
    os.makedirs(os.path.join(base_dir, 'all_images'), exist_ok=True)

    # Download and process all images
    for species_name, search_terms in TREE_SPECIES.items():
        print(f"\nProcessing species: {species_name}")
        
        success = False
        attempts = 0
        max_attempts = 3
        
        while not success and attempts < max_attempts:
            if attempts > 0:
                print(f"Attempt {attempts + 1}/{max_attempts}")
            
            success = download_and_process_images(
                species_name, 
                search_terms, 
                base_dir, 
                target_count=50
            )
            
            if not success:
                print(f"Failed to get 50 images for {species_name}, retrying...")
                attempts += 1
        
        if not success:
            print(f"Failed to get 50 images for {species_name} after {max_attempts} attempts")
            exit(1)
        
        print(f"Successfully processed 50 images for {species_name}")

    print("\nDataset creation completed!") 