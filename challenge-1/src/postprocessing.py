"""
Postprocessing module for soil classification challenge.

This module contains functions for prediction and result processing
for the soil classification task.
"""

import torch
import numpy as np
from typing import List, Dict, Union, Tuple
from pathlib import Path

def predict_image(image_path: Union[str, Path], 
                 model, 
                 preprocess, 
                 clf, 
                 text_embeddings: Dict[str, torch.Tensor], 
                 classes: List[str],
                 device: str = "cuda") -> str:
    """
    Predict soil class for a single image using hybrid approach.
    
    Args:
        image_path: Path to the image file
        model: Pre-trained model for feature extraction
        preprocess: Preprocessing function for the model
        clf: Trained classifier model
        text_embeddings: Dictionary of text embeddings for zero-shot classification
        classes: List of class names
        device: Device to use for computation ('cuda' or 'cpu')
        
    Returns:
        Predicted class name
    """
    from .preprocessing import get_image_embeddings
    
    # Get embeddings for logistic regression
    img_emb = get_image_embeddings(model, preprocess, [image_path], batch_size=1, device=device)
    probe_pred = clf.predict_proba(img_emb)
    
    # Get embeddings for zero-shot classification
    from PIL import Image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        zero_shot_probs = []
        for cls in classes:
            text_features = text_embeddings[cls].to(device)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            zero_shot_probs.append((image_features @ text_features.T).item())
        zero_shot_probs = torch.softmax(torch.tensor(zero_shot_probs), dim=0).numpy()
    
    # Weighted ensemble (70% logistic regression, 30% zero-shot)
    combined_probs = 0.7*probe_pred + 0.3*zero_shot_probs
    return classes[np.argmax(combined_probs)]

def batch_predict(image_paths: List[Union[str, Path]], 
                 model, 
                 preprocess, 
                 clf, 
                 text_embeddings: Dict[str, torch.Tensor], 
                 classes: List[str],
                 device: str = "cuda",
                 batch_size: int = 8) -> List[str]:
    """
    Predict soil classes for multiple images.
    
    Args:
        image_paths: List of paths to image files
        model: Pre-trained model for feature extraction
        preprocess: Preprocessing function for the model
        clf: Trained classifier model
        text_embeddings: Dictionary of text embeddings for zero-shot classification
        classes: List of class names
        device: Device to use for computation ('cuda' or 'cpu')
        batch_size: Number of images to process in each batch
        
    Returns:
        List of predicted class names
    """
    predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_predictions = [predict_image(path, model, preprocess, clf, 
                                          text_embeddings, classes, device) 
                            for path in batch_paths]
        predictions.extend(batch_predictions)
    return predictions

def create_submission(test_df, predictions: List[str], output_path: str = "submission.csv") -> str:
    """
    Create a submission file from predictions.
    
    Args:
        test_df: DataFrame containing test image IDs
        predictions: List of predicted class names
        output_path: Path to save the submission file
        
    Returns:
        Path to the saved submission file
    """
    import pandas as pd
    
    submission_df = test_df.copy()
    submission_df["soil_type"] = predictions
    submission_df.to_csv(output_path, index=False)
    
    # Print prediction distribution
    print(f"Prediction distribution:\n{submission_df['soil_type'].value_counts()}")
    
    return output_path
