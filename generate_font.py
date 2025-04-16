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

# Fira Code font configuration
FONT_PATH = "FiraCode-Retina.ttf"
FONT_SIZE = 15  # Reduced from 30
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Calculate dimensions based on the font
CHAR_WIDTH = font.getbbox("M")[2]
LINE_HEIGHT = math.ceil(font.getbbox("Mjpqy")[3] * 0.9)
CHAR_HEIGHT = LINE_HEIGHT

# Sheet dimensions
SHEET_WIDTH = 240  # Reduced from 480
CHARS_PER_ROW = SHEET_WIDTH // CHAR_WIDTH  # Integer division rounds down
NUM_ROWS = 5
# Force exact height to match generated BMPs (80 pixels)
SHEET_HEIGHT = 80  # Fixed height to match generated images
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

    # Format string with line breaks to fit width
    formatted_text = ""
    line_chars = 0
    for char in string:
        if line_chars >= CHARS_PER_ROW:
            formatted_text += "\n"
            line_chars = 0
        formatted_text += char
        line_chars += 1

    # Draw the entire string at once with proper line breaks
    draw.text((0, 0), formatted_text, font=font, fill=0)  # Black text

    # Convert the image to a numpy array
    img_array = np.array(temp_img)

    # Convert to grayscale values where black pixels (0) become 0.0 and white pixels (255) become 1.0
    # with intermediate grayscale values preserved
    grayscale_array = (img_array / 255.0).astype(np.float32)

    # Copy to target sheet
    target_sheet[:] = grayscale_array

    return target_sheet

# Helper function to generate a chunk of samples
def _generate_dataset_chunk(chunk_id, start_idx, end_idx, min_length, samples_dir, num_samples_to_save):
    """Generate a chunk of dataset samples with a deterministic seed derived from chunk_id"""
    # Set seed based on global seed and chunk ID for determinism
    chunk_seed = SEED + chunk_id
    random.seed(chunk_seed)
    np.random.seed(chunk_seed)

    chunk_size = end_idx - start_idx

    # Pre-allocate arrays for this chunk
    chunk_inputs = []
    chunk_strings = []
    chunk_targets = np.zeros((chunk_size, SHEET_HEIGHT, SHEET_WIDTH), dtype=np.float32)

    # Generate samples for this chunk
    for i in range(chunk_size):
        sample_idx = start_idx + i

        # Generate a random string that fits in the sheet
        length = random.randint(min_length, MAX_CHARS_PER_SHEET)
        string = generate_random_string(length)
        chunk_strings.append(string)

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        chunk_inputs.append(ascii_codes)

        # Place string on sheet using shared function
        place_string_on_sheet(string, chunk_targets[i])

        # Save this sample as an image if requested
        if sample_idx < num_samples_to_save:
            # Ensure directory exists
            os.makedirs(samples_dir, exist_ok=True)
            # Convert binary sheet to image and save it
            filename = f"{samples_dir}/input_{sample_idx}_{string[:20]}.bmp"
            binary_array_to_image(chunk_targets[i], output_path=filename)

    return chunk_inputs, chunk_targets

# Create a dataset of text sheets and save sample images to a folder
def create_string_dataset(num_samples=1000, min_length=20, samples_dir="train_input",
                         num_samples_to_save=10, num_workers=None):
    """Create a dataset of text sheets with consistent character grid dimensions using parallel processing."""
    import multiprocessing

    # Determine number of workers (default to CPU count)
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Ensure we don't use more workers than samples
    num_workers = min(num_workers, num_samples)

    print(f"Generating {num_samples} samples using {num_workers} parallel workers...")

    # Calculate chunk sizes
    chunk_size = num_samples // num_workers
    remainder = num_samples % num_workers

    # Create chunks with balanced sizes
    chunks = []
    start_idx = 0
    for i in range(num_workers):
        # Add one extra to the first 'remainder' chunks
        size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + size
        chunks.append((i, start_idx, end_idx))
        start_idx = end_idx

    os.makedirs(samples_dir, exist_ok=True)
    print(f"Saving {min(num_samples, num_samples_to_save)} sample sheets to {samples_dir}/")

    # Generate chunks in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            _generate_dataset_chunk,
            [(chunk_id, start, end, min_length, samples_dir, num_samples_to_save)
             for chunk_id, start, end in chunks]
        )

    # Combine results from all chunks
    all_inputs = []
    all_targets = []

    for chunk_inputs, chunk_targets in results:
        all_inputs.extend(chunk_inputs)
        all_targets.append(chunk_targets)

    # Combine all target arrays
    all_targets = np.concatenate(all_targets, axis=0)

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
        num_samples_to_save=20,  # Save 20 samples for reference
        num_workers=32
    )


    print("Dataset generation complete. Check the train_input/ directory.")
