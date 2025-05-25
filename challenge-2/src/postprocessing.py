"""
Author: Annam.ai IIT Ropar
Team Name: RAV
Team Members: VIGNESH J, ASHWATH VINODKUMAR, RAHUL BHARGAV TALLADA
Leaderboard Rank: 13
"""

# Postprocessing utilities for soil classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def save_predictions(predictions, image_ids, output_file="binary_soil_submission.csv"):
    """
    Save predictions to a CSV file in the required format
    
    Args:
        predictions (list or array): Binary predictions (0 or 1)
        image_ids (list): List of image IDs corresponding to predictions
        output_file (str): Path to save the output CSV
        
    Returns:
        Path to the saved file
    """
    # Create DataFrame
    submission_df = pd.DataFrame({
        "image_id": image_ids,
        "is_soil": predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return output_file

def analyze_predictions(predictions):
    """
    Analyze the distribution of predictions
    
    Args:
        predictions (list or array): Binary predictions (0 or 1)
        
    Returns:
        dict: Statistics about predictions
    """
    predictions = np.array(predictions)
    
    # Calculate statistics
    total = len(predictions)
    soil_count = np.sum(predictions)
    non_soil_count = total - soil_count
    soil_percentage = (soil_count / total) * 100
    
    stats = {
        "total": total,
        "soil_count": int(soil_count),
        "non_soil_count": int(non_soil_count),
        "soil_percentage": soil_percentage
    }
    
    # Print summary
    print(f"Prediction distribution:")
    print(f"Soil (1): {soil_count} images")
    print(f"Non-soil (0): {non_soil_count} images")
    print(f"Percentage classified as soil: {soil_percentage:.1f}%")
    
    return stats

def plot_confusion_matrix(y_true, y_pred, classes=['Non-Soil', 'Soil'], normalize=False):
    """
    Plot confusion matrix for binary classification
    
    Args:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        classes (list): Class names
        normalize (bool): Whether to normalize the confusion matrix
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show class labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def generate_classification_report(y_true, y_pred):
    """
    Generate and print classification report
    
    Args:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        
    Returns:
        dict: Classification metrics
    """
    report = classification_report(y_true, y_pred, target_names=['Non-Soil', 'Soil'], output_dict=True)
    print(classification_report(y_true, y_pred, target_names=['Non-Soil', 'Soil']))
    return report
