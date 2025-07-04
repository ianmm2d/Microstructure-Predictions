import os
import numpy as np
import torch

def get_image_data(npy_dir: str) -> tuple[torch.Tensor, list]:
    """
    Loads images from a directory containing .npy files and returns them as a tensor.
    
    Args:
        npy_dir (str): Directory containing .npy files of images.
    
    Returns:
        torch.Tensor: A tensor containing the stacked images.
        list: A list of file names corresponding to the image sample.
    """
    
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])

    # Preload and stack all images
    images = np.stack([np.load(os.path.join(npy_dir, f)) for f in npy_files])
    images = images.astype(np.float32)
    
    # Add channel dimension: (N, 1, H, W)
    images = np.expand_dims(images, axis=1)
    
    return torch.tensor(images), npy_files