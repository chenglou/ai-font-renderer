"""
Helper functions for the AI Font Renderer project.

This file contains utility functions for:
- Rendering strings as bitmap images
- Converting binary arrays to images
- Saving and loading model weights
- Loading training data
"""

import torch
import os
import numpy as np
from PIL import Image
import torch.utils.data as data

# Model weight filename
MODEL_FILENAME = "font_renderer.pth"

def binary_array_to_image(binary_array, output_path=None):
    """
    Convert a grayscale array (where 0=black, 1=white) to a PIL Image.

    Args:
        binary_array (numpy.ndarray): A 2D array with values 0.0-1.0 where 0.0 represents black and 1.0 represents white
        output_path (str, optional): If provided, save the image to this path

    Returns:
        PIL.Image: The converted image object
    """
    # Convert grayscale array to image format (255 for white, 0 for black)
    # Scale from 0.0-1.0 to 0-255
    img = (binary_array * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img)

    # Save if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        pil_img.save(output_path, "BMP")

    return pil_img

def render_strings(model, strings, output_dir, sheet_height, sheet_width, device):
    """Render a list of strings as BMP images"""
    os.makedirs(output_dir, exist_ok=True)

    for idx, string in enumerate(strings):
        # Cap string length to model's max_length
        if len(string) > model.max_length:
            string = string[:model.max_length]
            print(f"Warning: String truncated to {model.max_length} characters: {string}")

        # Convert to ASCII codes and pad
        ascii_codes = [ord(c) for c in string]
        if len(ascii_codes) < model.max_length:
            ascii_codes = ascii_codes + [0] * (model.max_length - len(ascii_codes))

        # Get model prediction
        x = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            sheet = model(x).squeeze(0)  # shape will be [SHEET_HEIGHT, SHEET_WIDTH]

        # Convert to numpy if needed
        if isinstance(sheet, torch.Tensor):
            sheet = sheet.detach().cpu().numpy()

        # Convert binary array to image and save
        filename = f"{output_dir}/string_{idx}.bmp"
        binary_array_to_image(sheet, output_path=filename)

    print(f"Saved {len(strings)} rendered strings to {output_dir}/")

def save_model(model, filename=MODEL_FILENAME):
    """Save model weights to a file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model_class, max_length, filename=MODEL_FILENAME, device=None):
    """
    Load model weights from a file
    
    Args:
        model_class: The model class to instantiate
        max_length: The maximum character length for the model
        filename: Path to the saved model weights
        device: The device to load the model on (cpu, cuda, mps)
        
    Returns:
        Loaded model in evaluation mode
    """
    # Initialize model with the provided parameters
    model = model_class(max_length=max_length)
    
    # Determine device if not provided
    if device is None:
        device = torch.device('cpu')
    
    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filename}")
    return model

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

def load_string_dataset(data_dir="train_input", num_samples=50000, sheet_height=80, sheet_width=240):
    """
    Load dataset from bitmap images and text file.

    Args:
        data_dir (str): Directory containing the bitmap images and text file
        num_samples (int): Number of samples to load
        sheet_height (int): Height of the bitmap sheets
        sheet_width (int): Width of the bitmap sheets

    Returns:
        torch.utils.data.TensorDataset: Dataset containing inputs and targets
    """
    print(f"Loading {num_samples} samples from {data_dir}...")

    # Initialize arrays to store data
    all_inputs = []
    all_targets = np.zeros((num_samples, sheet_height, sheet_width), dtype=np.float32)

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

    print(f"Dataset loading complete: {num_samples} samples with dimensions {sheet_height}x{sheet_width}")

    return data.TensorDataset(inputs_tensor, targets_tensor)