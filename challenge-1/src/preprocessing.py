"""
Preprocessing module for soil classification challenge.

This module contains functions for loading and preprocessing soil images
for the classification task.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Tuple

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from the specified path and convert to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object in RGB format
    """
    return Image.open(image_path).convert("RGB")

def batch_load_images(image_paths: List[Union[str, Path]], batch_size: int = 8) -> List[Image.Image]:
    """
    Load multiple images in batches.
    
    Args:
        image_paths: List of paths to image files
        batch_size: Number of images to load in each batch
        
    Returns:
        List of PIL Image objects
    """
    images = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [load_image(p) for p in batch_paths]
        images.extend(batch_images)
    return images

def get_image_embeddings(model, preprocess, image_paths: List[Union[str, Path]], 
                        batch_size: int = 8, device: str = "cuda") -> np.ndarray:
    """
    Generate embeddings for images using a pre-trained model.
    
    Args:
        model: Pre-trained model for feature extraction
        preprocess: Preprocessing function for the model
        image_paths: List of paths to image files
        batch_size: Number of images to process in each batch
        device: Device to use for computation ('cuda' or 'cpu')
        
    Returns:
        Array of image embeddings
    """
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch = torch.stack([preprocess(load_image(p)) for p in batch_paths])
        
        with torch.no_grad():
            batch = batch.to(device)
            batch_emb = model.encode_image(batch)
            embeddings.append(batch_emb.cpu().numpy())
        
        # Explicit memory cleanup
        del batch, batch_emb
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return np.concatenate(embeddings)

def generate_text_embeddings(model, class_prompts: dict, device: str = "cuda") -> dict:
    """
    Generate text embeddings for class prompts.
    
    Args:
        model: Pre-trained model for feature extraction
        class_prompts: Dictionary mapping class names to lists of text prompts
        device: Device to use for computation ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping class names to text embeddings
    """
    import open_clip
    
    with torch.no_grad():
        text_embeddings = {}
        for cls, prompts in class_prompts.items():
            embeddings = []
            for prompt in prompts:
                text = open_clip.tokenize([prompt]).to(device)
                embeddings.append(model.encode_text(text))
            text_embeddings[cls] = torch.mean(torch.cat(embeddings), dim=0, keepdim=True)
    
    return text_embeddings
