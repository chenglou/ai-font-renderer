"""
This file implements utilities for generating font bitmap training data using a font

The module provides functions for:
- Generating random strings
- Rendering characters using a font
- Creating training datasets with bitmap font images
- Saving sample images

Usage:
  - Run directly to generate samples: python generate_font.py
  - Import and use in model training: import generate_font
"""

import random
import os
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.utils.data as data

def binary_array_to_image(binary_array, output_path=None):
    """
    Convert a binary array (where 1=black, 0=white) to a PIL Image.

    Args:
        binary_array (numpy.ndarray): A 2D array with values 0.0-1.0 where 1.0 represents black
        output_path (str, optional): If provided, save the image to this path

    Returns:
        PIL.Image: The converted image object
    """
    # Convert binary array to image format (255 for white, 0 for black)
    img = np.ones(binary_array.shape, dtype=np.uint8) * 255
    img[binary_array >= 0.5] = 0  # Set black pixels where binary value is >= 0.5

    # Convert to PIL Image
    pil_img = Image.fromarray(img)

    # Save if path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        pil_img.save(output_path, "BMP")

    return pil_img

# Fira Code font configuration
FONT_PATH = "FiraCode-Retina.ttf"
FONT_SIZE = 30
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Calculate dimensions based on the font
CHAR_WIDTH = font.getbbox("M")[2]
LINE_HEIGHT = math.ceil(font.getbbox("Mjpqy")[3] * 0.9)
CHAR_HEIGHT = LINE_HEIGHT

# Sheet dimensions
SHEET_WIDTH = 480  # Keep this fixed as specified
CHARS_PER_ROW = SHEET_WIDTH // CHAR_WIDTH  # Integer division rounds down
NUM_ROWS = 5
SHEET_HEIGHT = math.ceil(LINE_HEIGHT * NUM_ROWS)  # Calculate height from line height
MAX_ROWS = SHEET_HEIGHT // CHAR_HEIGHT
MAX_CHARS_PER_SHEET = CHARS_PER_ROW * MAX_ROWS

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Generate random strings (uppercase and space only)
def generate_random_string(length):
    available_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    return ''.join(random.choice(available_chars) for _ in range(length))

# Place string characters as bitmap on a target sheet using the TTF font
def place_string_on_sheet(string, target_sheet):
    """
    Places a string on a target sheet as bitmaps using the FiraCode font.

    Args:
        string (str): The string to render
        target_sheet (numpy.ndarray): Target array to place characters on

    Returns:
        numpy.ndarray: Updated target sheet with rendered string
    """
    # Create a temporary image to render the text
    temp_img = Image.new('L', (SHEET_WIDTH, SHEET_HEIGHT), 255)  # White background
    draw = ImageDraw.Draw(temp_img)

    char_idx = 0
    for row in range(MAX_ROWS):
        if char_idx >= len(string):
            break

        y_pos = row * CHAR_HEIGHT

        for col in range(CHARS_PER_ROW):
            if char_idx >= len(string):
                break

            x_pos = col * CHAR_WIDTH

            # Draw the character at the position
            if string[char_idx] != '':
                draw.text((x_pos, y_pos), string[char_idx], font=font, fill=0)  # Black text

            char_idx += 1

    # Convert the image to a numpy array
    img_array = np.array(temp_img)

    # Convert to binary where black pixels (0) become 1 and white pixels (255) become 0
    # This maintains compatibility with the existing model (1 = black)
    binary_array = (img_array < 128).astype(np.float32)

    # Copy to target sheet
    target_sheet[:] = binary_array

    return target_sheet

# Create a dataset of text sheets and save sample images to a folder
def create_string_dataset(num_samples=1000, min_length=20, samples_dir="train_input", num_samples_to_save=10):
    """Create a dataset of text sheets with consistent character grid dimensions."""
    # Reset random seed for reproducible dataset generation
    random.seed(SEED)

    # Pre-allocate arrays for better performance
    all_inputs = []
    all_strings = []  # Store generated strings for reference
    all_targets = np.zeros((num_samples, SHEET_HEIGHT, SHEET_WIDTH), dtype=np.float32)

    os.makedirs(samples_dir, exist_ok=True)
    print(f"Saving {min(num_samples, num_samples_to_save)} sample sheets to {samples_dir}/")

    for sample_idx in range(num_samples):
        # Generate a random string that fits in the sheet
        length = random.randint(min_length, MAX_CHARS_PER_SHEET)
        string = generate_random_string(length)
        all_strings.append(string)

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        all_inputs.append(ascii_codes)

        # Place string on sheet using shared function
        place_string_on_sheet(string, all_targets[sample_idx])

        # Save this sample as an image if requested
        if sample_idx < num_samples_to_save:
            # Convert binary sheet to image and save it
            filename = f"{samples_dir}/input_{sample_idx}_{string[:20]}.bmp"
            binary_array_to_image(all_targets[sample_idx], output_path=filename)

    # Pad sequences to max_length
    max_len = max(len(s) for s in all_inputs)
    padded_inputs = np.zeros((num_samples, max_len), dtype=np.int64)

    for i, codes in enumerate(all_inputs):
        # Pad inputs with zeros
        padded_inputs[i, :len(codes)] = codes

    # Convert to tensors
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

    print(f"Dataset creation complete: {num_samples} samples with dimensions {SHEET_HEIGHT}x{SHEET_WIDTH}")

    return data.TensorDataset(inputs_tensor, targets_tensor)

def generate_challenging_patterns():
    """Generate a dataset with challenging patterns for validation testing"""

    patterns = [
        "IIIIIIIIIIIIIIIIIIII",  # Repeating I's
        "WWWWWWWWWWWWWWWWWWWW",  # Repeating W's
        "IIIII IIIII IIIII IIIII",  # Groups of I's with spaces
        "WWWWW WWWWW WWWWW WWWWW",  # Groups of W's with spaces
        "IWIWIWIWIWIWIWIWIWIWI",  # Alternating I and W pattern
        "                     ",  # Just spaces
    ]

    # Prepare tensors for patterns
    pattern_inputs = []
    pattern_targets = []

    # Create dataset from patterns
    for idx, pattern in enumerate(patterns):
        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in pattern]
        # Pad to max_length
        if len(ascii_codes) < MAX_CHARS_PER_SHEET:
            ascii_codes = ascii_codes + [0] * (MAX_CHARS_PER_SHEET - len(ascii_codes))

        # Create input tensor
        pattern_input = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0)
        pattern_inputs.append(pattern_input)

        # Create target bitmap
        target = np.zeros((SHEET_HEIGHT, SHEET_WIDTH), dtype=np.float32)

        # Place pattern on sheet using shared function
        place_string_on_sheet(pattern, target)

        pattern_targets.append(torch.tensor(target, dtype=torch.float32).unsqueeze(0))

        # Save these pattern samples for visualization
        os.makedirs("train_input", exist_ok=True)
        filename = f"train_input/test_{idx}_{pattern.replace(' ', '_')}.bmp"
        binary_array_to_image(target, output_path=filename)

    # Concatenate and return dataset
    pattern_inputs = torch.cat(pattern_inputs, dim=0)
    pattern_targets = torch.cat(pattern_targets, dim=0)
    return data.TensorDataset(pattern_inputs, pattern_targets)

def create_dataset_metadata(samples_dir="train_input"):
    """Create a metadata file with information about the dataset"""
    os.makedirs(samples_dir, exist_ok=True)

    with open(f"{samples_dir}/dataset_metadata.txt", "w") as f:
        f.write("AI Font Renderer Dataset\n")
        f.write("========================\n\n")
        f.write(f"Font: {FONT_PATH}\n")
        f.write(f"Font size: {FONT_SIZE}\n")
        f.write(f"Character dimensions: {CHAR_WIDTH}x{CHAR_HEIGHT}\n")
        f.write(f"Sheet dimensions: {SHEET_WIDTH}x{SHEET_HEIGHT}\n")
        f.write(f"Characters per row: {CHARS_PER_ROW}\n")
        f.write(f"Maximum rows: {MAX_ROWS}\n")
        f.write(f"Maximum characters per sheet: {MAX_CHARS_PER_SHEET}\n\n")
        f.write("Format: Each sample consists of an image (.bmp) and its corresponding text (.txt)\n")
        f.write("All samples use the same seed for reproducibility (SEED=42)\n")

if __name__ == "__main__":
    print("Generating font bitmap training data...")

    # Create the dataset metadata
    create_dataset_metadata()

    # Create the main training dataset
    dataset = create_string_dataset(
        num_samples=5000,  # Generate 5000 samples
        min_length=10,
        samples_dir="train_input",
        num_samples_to_save=20  # Save 20 samples for reference
    )

    # Generate and save challenging test patterns
    generate_challenging_patterns()

    print("Dataset generation complete. Check the train_input/ directory.")
