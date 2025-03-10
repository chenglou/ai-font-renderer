import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import multiprocessing as mp
import time

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Directory for output
OUTPUT_DIR = "train_monospace_input"

# Sheet layout configuration
CHARS_PER_ROW = 20  # Number of characters per row
NUM_ROWS = 5        # Number of rows in each sheet

# Calculate dimensions based on FiraCode-Retina with font size 40
# We precompute these to ensure consistency
FONT_SIZE = 40
FONT_PATH = "FiraCode-Retina.ttf"

# Initialize font to get metrics
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
CHAR_WIDTH = font.getbbox("M")[2]
LINE_HEIGHT = math.ceil(font.getbbox("Mjpqy")[3] * 0.9)  # 0.9 to reduce line spacing

# Calculate sheet dimensions
SHEET_WIDTH = math.ceil(CHAR_WIDTH * CHARS_PER_ROW)
SHEET_HEIGHT = math.ceil(LINE_HEIGHT * NUM_ROWS)
MAX_CHARS_PER_SHEET = CHARS_PER_ROW * NUM_ROWS

def generate_random_string(length, local_random=None):
    """Generate a random string of uppercase letters and spaces."""
    if local_random is None:
        local_random = random
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    return ''.join(local_random.choice(characters) for _ in range(length))

def render_string_to_bitmap(string, output_path):
    """Render a string to a bitmap image using PIL, rendering entire lines at once."""
    # Create a white background image
    image = Image.new('L', (SHEET_WIDTH, SHEET_HEIGHT), color=255)
    draw = ImageDraw.Draw(image)

    # Use the globally defined font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Split the string into rows
    char_idx = 0
    rows = []

    while char_idx < len(string):
        # Get characters for this row (limited by CHARS_PER_ROW)
        row_chars = string[char_idx:char_idx + CHARS_PER_ROW]
        rows.append(row_chars)
        char_idx += len(row_chars)

        # Break if we've filled all rows
        if len(rows) >= NUM_ROWS:
            break

    # Render each row as a complete string
    for row_idx, row_text in enumerate(rows):
        # Position the row
        y = row_idx * LINE_HEIGHT

        # Draw the entire row at once
        draw.text((0, y), row_text, font=font, fill=0)

    # Save the image
    image.save(output_path, "BMP")
    return image

def process_batch(batch_data):
    """Process a batch of samples at once within a single process."""
    batch_start_idx, batch_size, min_length, seed_offset = batch_data

    # Initialize local random state for this batch
    local_random = random.Random(SEED + seed_offset)

    # Process all samples in this batch
    for i in range(batch_size):
        sample_idx = batch_start_idx + i

        # Generate a random string that fits in the sheet
        length = local_random.randint(min_length, MAX_CHARS_PER_SHEET)
        string = generate_random_string(length, local_random)

        # Render and save the image
        filename = f"{OUTPUT_DIR}/{sample_idx}_{string[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(string, filename)

    return batch_start_idx

def process_special_patterns(patterns_batch):
    """Process a batch of special patterns at once."""
    results = []
    for idx, pattern in patterns_batch:
        filename = f"{OUTPUT_DIR}/special_{idx}_{pattern[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(pattern, filename)
        results.append(idx)
    return results

def create_dataset_parallel(num_samples=100, min_length=10, num_processes=None):
    """Create a dataset of text sheets with the specified font using parallel processing with batching."""
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Calculate optimal batch size based on number of samples and processes
    # Each process will handle multiple samples at once
    batch_size = max(10, math.ceil(num_samples / num_processes))
    num_batches = math.ceil(num_samples / batch_size)

    print(f"Creating {num_samples} sample sheets in {OUTPUT_DIR}/")
    print(f"Using {num_processes} processes with {num_batches} batches (approx. {batch_size} samples per batch)")

    # Prepare batch data
    batch_data = []
    for b in range(num_batches):
        start_idx = b * batch_size
        # Adjust batch size for the last batch if needed
        current_batch_size = min(batch_size, num_samples - start_idx)
        batch_data.append((start_idx, current_batch_size, min_length, b))

    # Special test patterns
    special_patterns = [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "IIIIIIIIIIIIIIIIIIII",  # Repeating narrow character
        "WWWWWWWWWWWWWWWWWWWW",  # Repeating wide character
        "ALTERNATING CASE TEST   SPACES",
        "IIIII IIIII IIIII IIIII",  # Groups with spaces
        "WWWWW WWWWW WWWWW WWWWW",  # Groups with spaces
        "IWIWIWIWIWIWIWIWIWIWI",  # Alternating pattern
    ]

    # Create a single batch of special patterns
    special_batch = list(enumerate(special_patterns))

    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        # Process regular sample batches in parallel
        print(f"Generating random samples in parallel...")
        results = list(pool.imap_unordered(process_batch, batch_data))

        # Process special patterns
        print("Generating special test patterns...")
        special_results = pool.apply(process_special_patterns, (special_batch,))

    elapsed_time = time.time() - start_time
    print(f"Dataset creation complete! Processed {num_samples + len(special_patterns)} samples in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Error: Font file {FONT_PATH} not found!")
        exit(1)

    # Print the configuration only once in the main process
    print(f"Sheet configuration:")
    print(f"- Font size: {FONT_SIZE}px")
    print(f"- Character width: {CHAR_WIDTH:.2f}px")
    print(f"- Line height: {LINE_HEIGHT}px")
    print(f"- Sheet dimensions: {SHEET_WIDTH}x{SHEET_HEIGHT} pixels")
    print(f"- Characters per sheet: {MAX_CHARS_PER_SHEET}")

    # Create the dataset with 5000 samples in parallel for better training
    create_dataset_parallel(num_samples=5000)
