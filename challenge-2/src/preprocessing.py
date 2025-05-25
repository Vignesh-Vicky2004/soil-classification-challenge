"""
Author: Annam.ai IIT Ropar
Team Name: RAV
Team Members: VIGNESH J, ASHWATH VINODKUMAR, RAHUL BHARGAV TALLADA
Leaderboard Rank: 13
"""

# Preprocessing utilities for soil classification

import numpy as np
from PIL import Image
import torch
from pathlib import Path

def load_and_preprocess_image(image_path, transform=None):
    """
    Load and preprocess an image for model input
    
    Args:
        image_path (str or Path): Path to the image file
        transform (callable, optional): Transformation to apply to the image
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations if provided
        if transform is not None:
            image = transform(image)
            
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def batch_preprocess_images(image_paths, transform, batch_size=8, device="cuda"):
    """
    Preprocess a batch of images to prevent memory issues
    
    Args:
        image_paths (list): List of image paths
        transform (callable): Transformation to apply to images
        batch_size (int): Size of batches to process
        device (str): Device to use for processing
        
    Returns:
        List of preprocessed image tensors
    """
    all_processed = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch = []
        
        for path in batch_paths:
            img = load_and_preprocess_image(path, transform)
            if img is not None:
                batch.append(img)
        
        if batch:
            batch_tensor = torch.stack(batch)
            all_processed.append(batch_tensor)
            
    return all_processed

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit length
    
    Args:
        embeddings (torch.Tensor): Embeddings to normalize
        
    Returns:
        Normalized embeddings
    """
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

def prepare_data_paths(data_dir, csv_file, image_column="image_id"):
    """
    Prepare full image paths from a CSV file
    
    Args:
        data_dir (str or Path): Base directory containing images
        csv_file (str or Path): Path to CSV file with image IDs
        image_column (str): Column name containing image IDs
        
    Returns:
        List of full image paths
    """
    import pandas as pd
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create full paths
    data_dir = Path(data_dir)
    image_paths = [data_dir / img_id for img_id in df[image_column]]
    
    return image_paths, df
