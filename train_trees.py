"""
Tree Species Classification Model Training Script
----------------------------------------------
This script implements a two-phase deep learning model for tree species classification.
It uses transfer learning with EfficientNetB0 as the base model and implements various
techniques to improve performance, including:
- Two-phase training (feature extraction and fine-tuning)
- Mixup augmentation
- Advanced data augmentation
- Learning rate scheduling
- Early stopping with model checkpointing

Author: Jaka Kus
Date: February 2024
License: MIT
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import warnings
import shutil
warnings.filterwarnings("ignore")

def split_dataset(base_dir, train_size=0.7, val_size=0.15):
    """Split the dataset into train, validation and test sets.
    
    Args:
        base_dir (str): Base directory containing the dataset
        train_size (float): Proportion of data for training (default: 0.7)
        val_size (float): Proportion of data for validation (default: 0.15)
        
    Note: The remaining proportion (1 - train_size - val_size) will be used for testing.
    """
    # Create directories for splits
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)

    # Process each class
    all_images_dir = os.path.join(base_dir, 'all_images')
    for class_name in os.listdir(all_images_dir):
        class_dir = os.path.join(all_images_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class directories in each split
        for split in splits:
            os.makedirs(os.path.join(base_dir, split, class_name))

        # Get all images
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        # First split: separate training set
        train_imgs, temp_imgs = train_test_split(images, train_size=train_size, random_state=42)
        
        # Second split: separate validation and test from the remaining data
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_size/(1-train_size), random_state=42)

        # Copy images to their respective directories
        for img in train_imgs:
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(base_dir, 'train', class_name, img)
            )
        
        for img in val_imgs:
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(base_dir, 'validation', class_name, img)
            )
        
        for img in test_imgs:
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(base_dir, 'test', class_name, img)
            )

    print("Dataset split completed!")
    # Print statistics
    for split in splits:
        total = sum(len(os.listdir(os.path.join(base_dir, split, d))) 
                   for d in os.listdir(os.path.join(base_dir, split)))
        print(f"{split} set: {total} images")

def create_data_generators():
    """Create and return data generators for training and validation.
    
    Returns:
        tuple: (train_datagen, val_datagen) containing:
            - train_datagen: ImageDataGenerator with augmentation for training
            - val_datagen: ImageDataGenerator for validation/testing
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    return train_datagen, val_datagen

def mixup_data(x, y, alpha=0.2):
    """Perform mixup augmentation on the input data and their labels.
    
    Args:
        x (tensor): Input images
        y (tensor): Input labels
        alpha (float): Mixup interpolation strength (default: 0.2)
        
    Returns:
        tuple: (mixed_x, mixed_y) containing mixed images and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, indices)
    
    return mixed_x, mixed_y

class MixupGenerator:
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
    
    def flow(self):
        while True:
            x_batch, y_batch = next(self.generator)
            x_mixed, y_mixed = mixup_data(x_batch, y_batch, self.alpha)
            yield x_mixed, y_mixed

def create_model(num_classes, phase=1, steps_per_epoch=None):
    """Create and return the tree classification model.
    
    Args:
        num_classes (int): Number of tree species to classify
        phase (int): Training phase (1=feature extraction, 2=fine-tuning)
        steps_per_epoch (int, optional): Steps per epoch for learning rate scheduling
        
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    if phase == 1:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-3]:
            layer.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=(phase==2))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Return to simpler, proven architecture
    x = tf.keras.layers.Dense(384, 
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(192, 
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    initial_lr = 0.0005 if phase == 1 else 0.0000005
    if phase == 2 and steps_per_epoch is not None:
        # Remove the cosine decay schedule for Phase 2
        lr_schedule = initial_lr
    else:
        lr_schedule = initial_lr

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=initial_lr
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
        metrics=['accuracy', 
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def calculate_class_weights(directory):
    """Calculate class weights to handle class imbalance.
    
    Args:
        directory (str): Directory containing the class subdirectories
        
    Returns:
        dict: Class weights indexed by class indices
    """
    total_samples = 0
    class_counts = {}
    
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            count = len(os.listdir(class_dir))
            class_counts[class_name] = count
            total_samples += count
    
    class_weights = {}
    n_classes = len(class_counts)
    
    for class_name, count in class_counts.items():
        # Calculate base weight
        base_weight = (1 / count) * (total_samples / n_classes)
        class_weights[list(class_counts.keys()).index(class_name)] = base_weight
    
    return class_weights

def plot_sample_images(data_dir, num_samples=16):
    """Plot a grid of sample images from the dataset.
    
    Args:
        data_dir (str): Directory containing the image data
        num_samples (int): Number of samples to display (default: 16)
    """
    image_paths = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
    
    random_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(random_images):
            img = mpimg.imread(random_images[i])
            ax.imshow(img)
            ax.axis('off')
            class_name = os.path.basename(os.path.dirname(random_images[i]))
            ax.set_title(class_name, fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_data_distribution(data_dir, title):
    """Plot the distribution of images across different classes.
    
    Args:
        data_dir (str): Directory containing the class subdirectories
        title (str): Title for the plot
    """
    classes = []
    counts = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            classes.append(class_name)
            counts.append(len(os.listdir(class_dir)))
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot and save the normalized confusion matrix.
    
    Args:
        y_true (array): True class labels
        y_pred (array): Predicted class labels
        classes (list): List of class names
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, 
                annot=True, 
                xticklabels=classes,
                yticklabels=classes,
                cmap='vlag')
    plt.xlabel("Predicted Classes", fontsize=15)
    plt.ylabel("True Classes", fontsize=15)
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    # Create output directory for results
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split dataset
    base_dir = 'trees_dataset'
    split_dataset(base_dir, train_size=0.7, val_size=0.15)
    
    # Define data directories
    train_dir = os.path.join('trees_dataset', 'train')
    val_dir = os.path.join('trees_dataset', 'validation')
    test_dir = os.path.join('trees_dataset', 'test')
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dir)
    
    # Create data generators with balanced augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=50.0,
        fill_mode="nearest"
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    # Create image generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=True
    )
    
    # Create mixup generator with proven alpha
    mixup_train_generator = MixupGenerator(train_generator, alpha=0.3).flow()
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    # Phase 1: Initial training with more epochs
    print("\nPhase 1: Training with frozen base model...")
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes, phase=1)
    
    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(output_dir, 'phase1_best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'phase1_training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    history_phase1 = model.fit(
        mixup_train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        epochs=20,  # Increased from 8
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tuning with unfrozen layers
    print("\nPhase 2: Fine-tuning with unfrozen layers...")
    # Load the best model from Phase 1
    model = tf.keras.models.load_model(os.path.join(output_dir, 'phase1_best_model.keras'))
    # Create new model for Phase 2 with loaded weights
    new_model = create_model(num_classes, phase=2, steps_per_epoch=len(train_generator))
    new_model.set_weights(model.get_weights())
    model = new_model
    
    callbacks_phase2 = [
        ModelCheckpoint(
            os.path.join(output_dir, 'phase2_best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'phase2_training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    history_phase2 = model.fit(
        mixup_train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        epochs=15,  # Increased from 6
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Plot combined training history
    plt.figure(figsize=(15, 5))
    
    # Combine histories
    total_epochs = len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy'])
    combined_epochs = range(1, total_epochs + 1)
    phase1_epochs = len(history_phase1.history['accuracy'])
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['accuracy'], 'b-', label='Phase 1 Training')
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['val_accuracy'], 'b--', label='Phase 1 Validation')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['accuracy'], 'r-', label='Phase 2 Training')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['val_accuracy'], 'r--', label='Phase 2 Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['loss'], 'b-', label='Phase 1 Training')
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['val_loss'], 'b--', label='Phase 1 Validation')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['loss'], 'r-', label='Phase 2 Training')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['val_loss'], 'r--', label='Phase 2 Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['top_2_accuracy'], 'b-', label='Phase 1 Training')
    plt.plot(range(1, phase1_epochs + 1), history_phase1.history['val_top_2_accuracy'], 'b--', label='Phase 1 Validation')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['top_2_accuracy'], 'r-', label='Phase 2 Training')
    plt.plot(range(phase1_epochs + 1, total_epochs + 1), history_phase2.history['val_top_2_accuracy'], 'r--', label='Phase 2 Validation')
    plt.title('Top-2 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-2 Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_training_history.png'))
    plt.close()
    
    # Load the best model from phase 2 for evaluation
    best_model = tf.keras.models.load_model(os.path.join(output_dir, 'phase2_best_model.keras'))
    
    # Evaluate on test set
    test_loss, test_accuracy, test_top2_acc, test_precision, test_recall, test_auc = best_model.evaluate(test_generator)
    print(f'\nTest accuracy: {test_accuracy:.2%}')
    print(f'Test top-2 accuracy: {test_top2_acc:.2%}')
    print(f'Test precision: {test_precision:.2%}')
    print(f'Test recall: {test_recall:.2%}')
    print(f'Test AUC: {test_auc:.2%}')
    
    # Generate predictions for confusion matrix
    predictions = best_model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 15))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(cm, 
                annot=True, 
                xticklabels=list(test_generator.class_indices.keys()),
                yticklabels=list(test_generator.class_indices.keys()),
                cmap='vlag',
                fmt='.2f')
    plt.xlabel("Predicted Classes", fontsize=15)
    plt.ylabel("True Classes", fontsize=15)
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate and save classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=list(test_generator.class_indices.keys()),
        digits=3
    )
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main() 