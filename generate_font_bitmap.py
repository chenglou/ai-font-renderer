import os
import random
import math
import time
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp

# Configuration
SEED = 42
OUTPUT_DIR = "train_monospace_input"
FONT_PATH = "FiraCode-Retina.ttf"
FONT_SIZE = 40
CHARS_PER_ROW = 20
NUM_ROWS = 5

# Special test patterns
SPECIAL_PATTERNS = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "IIIIIIIIIIIIIIIIIIII",  # Repeating narrow character
    "WWWWWWWWWWWWWWWWWWWW",  # Repeating wide character
    "ALTERNATING CASE TEST   SPACES",
    "IIIII IIIII IIIII IIIII",  # Groups with spaces
    "WWWWW WWWWW WWWWW WWWWW",  # Groups with spaces
    "IWIWIWIWIWIWIWIWIWIWI",  # Alternating pattern
]

# Calculate dimensions (done once in main process)
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
CHAR_WIDTH = font.getbbox("M")[2]
LINE_HEIGHT = math.ceil(font.getbbox("Mjpqy")[3] * 0.9)
SHEET_WIDTH = math.ceil(CHAR_WIDTH * CHARS_PER_ROW)
SHEET_HEIGHT = math.ceil(LINE_HEIGHT * NUM_ROWS)
MAX_CHARS_PER_SHEET = CHARS_PER_ROW * NUM_ROWS

def render_string_to_bitmap(string, output_path):
    """Render a string to a bitmap image."""
    # Create image with white background
    image = Image.new('L', (SHEET_WIDTH, SHEET_HEIGHT), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    
    # Split into rows and render
    rows = []
    char_idx = 0
    
    while char_idx < len(string) and len(rows) < NUM_ROWS:
        row = string[char_idx:char_idx + CHARS_PER_ROW]
        rows.append(row)
        char_idx += len(row)
        
    for row_idx, text in enumerate(rows):
        y = row_idx * LINE_HEIGHT
        draw.text((0, y), text, font=font, fill=0)
    
    # Save the image
    image.save(output_path, "BMP")

def process_batch(batch_data):
    """Process a batch of samples at once."""
    start_idx, batch_size, min_length, seed_offset = batch_data
    
    # Each process gets its own random generator
    local_random = random.Random(SEED + seed_offset)
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    
    for i in range(batch_size):
        idx = start_idx + i
        
        # Generate random text
        length = local_random.randint(min_length, MAX_CHARS_PER_SHEET)
        text = ''.join(local_random.choice(characters) for _ in range(length))
        
        # Create and save bitmap
        filename = f"{OUTPUT_DIR}/{idx}_{text[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(text, filename)
    
    return batch_size

def process_special_patterns():
    """Process all special test patterns."""
    for idx, pattern in enumerate(SPECIAL_PATTERNS):
        filename = f"{OUTPUT_DIR}/special_{idx}_{pattern[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(pattern, filename)
    return len(SPECIAL_PATTERNS)

def create_dataset_parallel(num_samples=100, min_length=10, num_processes=None):
    """Create a font bitmap dataset using parallel processing."""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine processing configuration
    num_processes = mp.cpu_count() if num_processes is None else num_processes
    batch_size = max(10, math.ceil(num_samples / num_processes))
    num_batches = math.ceil(num_samples / batch_size)
    
    print(f"Creating {num_samples} samples in {OUTPUT_DIR}/")
    print(f"Using {num_processes} processes with {num_batches} batches (~{batch_size} samples/batch)")
    
    # Prepare batch data
    batches = []
    for b in range(num_batches):
        start_idx = b * batch_size
        size = min(batch_size, num_samples - start_idx)
        batches.append((start_idx, size, min_length, b))
    
    # Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Handle random samples
        print("Generating random samples...")
        results = pool.map(process_batch, batches)
        
        # Handle special patterns
        print("Generating special test patterns...")
        special_count = pool.apply(process_special_patterns)
    
    elapsed = time.time() - start_time
    print(f"Dataset creation complete! Generated {num_samples + len(SPECIAL_PATTERNS)} "
          f"samples in {elapsed:.2f} seconds")

if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Error: Font file {FONT_PATH} not found!")
        exit(1)
        
    # Print configuration
    print(f"Sheet configuration:")
    print(f"- Font size: {FONT_SIZE}px")
    print(f"- Character width: {CHAR_WIDTH:.2f}px")
    print(f"- Line height: {LINE_HEIGHT}px")
    print(f"- Sheet dimensions: {SHEET_WIDTH}x{SHEET_HEIGHT} pixels")
    print(f"- Characters per sheet: {MAX_CHARS_PER_SHEET}")
    
    # Create dataset with 5000 samples
    create_dataset_parallel(num_samples=5000)