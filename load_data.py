"""
This file implements utilities for loading font bitmap training data from disk.

The module provides functions for:
- Loading bitmap images from files
- Loading string data from a text file
- Creating a tensor dataset from the loaded data
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from generate_font import SHEET_HEIGHT, SHEET_WIDTH, MAX_CHARS_PER_SHEET
from generate_font import binary_array_to_image

def image_to_binary_array(image_path):
    """
    Convert a bitmap image to a binary numpy array.
    
    Args:
        image_path (str): Path to the bitmap image
        
    Returns:
        numpy.ndarray: A 2D array with values 0.0-1.0 where 0.0 represents black and 1.0 represents white
    """
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Convert to numpy array and normalize to 0.0-1.0
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array

def load_string_dataset(data_dir="train_input", num_samples=50000):
    """
    Load dataset from bitmap images and text file.
    
    Args:
        data_dir (str): Directory containing the bitmap images and text file
        num_samples (int): Number of samples to load
        
    Returns:
        torch.utils.data.TensorDataset: Dataset containing inputs and targets
    """
    print(f"Loading {num_samples} samples from {data_dir}...")
    
    # Initialize arrays to store data
    all_inputs = []
    all_targets = np.zeros((num_samples, SHEET_HEIGHT, SHEET_WIDTH), dtype=np.float32)
    
    # Load strings from data.txt
    strings_path = os.path.join(data_dir, "data.txt")
    with open(strings_path, 'r') as f:
        strings = f.read().splitlines()
    
    if len(strings) < num_samples:
        raise ValueError(f"Not enough strings in {strings_path}. Expected {num_samples}, got {len(strings)}")
    
    # Process each sample
    for i in range(num_samples):
        # Load the bitmap image
        image_path = os.path.join(data_dir, f"{i+1}.bmp")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Convert image to binary array
        all_targets[i] = image_to_binary_array(image_path)
        
        # Convert string to ASCII codes
        string = strings[i]
        ascii_codes = [ord(c) for c in string]
        all_inputs.append(ascii_codes)
    
    # Pad sequences to max_length
    max_len = max(len(s) for s in all_inputs)
    padded_inputs = np.zeros((num_samples, max_len), dtype=np.int64)
    
    for i, codes in enumerate(all_inputs):
        # Pad inputs with zeros
        padded_inputs[i, :len(codes)] = codes
    
    # Convert to tensors
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
    
    print(f"Dataset loading complete: {num_samples} samples with dimensions {SHEET_HEIGHT}x{SHEET_WIDTH}")
    
    return data.TensorDataset(inputs_tensor, targets_tensor)

if __name__ == "__main__":
    # Test loading a small subset of data
    dataset = load_string_dataset(num_samples=10)
    print(f"Successfully loaded {len(dataset)} samples")
    
    # Access a sample
    inputs, targets = dataset[0]
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    
    # Save the first sample as an image for verification
    binary_array_to_image(targets.numpy(), output_path="test_loaded_sample.bmp")
    print("Saved test sample to test_loaded_sample.bmp")