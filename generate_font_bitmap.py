import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

print(f"Sheet configuration:")
print(f"- Font size: {FONT_SIZE}px")
print(f"- Character width: {CHAR_WIDTH:.2f}px")
print(f"- Line height: {LINE_HEIGHT}px")
print(f"- Sheet dimensions: {SHEET_WIDTH}x{SHEET_HEIGHT} pixels")
print(f"- Characters per sheet: {MAX_CHARS_PER_SHEET}")

# No longer needed since we're using constants

def generate_random_string(length):
    """Generate a random string of uppercase letters and spaces."""
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    return ''.join(random.choice(characters) for _ in range(length))

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

def create_dataset(num_samples=100, min_length=10):
    """Create a dataset of text sheets with the specified font."""
    # Reset random seed for reproducible dataset generation
    random.seed(SEED)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Creating {num_samples} sample sheets in {OUTPUT_DIR}/")

    for sample_idx in range(num_samples):
        # Generate a random string that fits in the sheet
        length = random.randint(min_length, MAX_CHARS_PER_SHEET)
        string = generate_random_string(length)

        # Render and save the image
        filename = f"{OUTPUT_DIR}/input_{sample_idx}_{string[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(string, filename)

        # Progress indicator for large datasets
        if (sample_idx + 1) % 10 == 0:
            print(f"Created {sample_idx + 1} samples...")

    # Generate some special test patterns
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

    print("Generating special test patterns...")
    for idx, pattern in enumerate(special_patterns):
        filename = f"{OUTPUT_DIR}/special_{idx}_{pattern[:20].replace(' ', '_')}.bmp"
        render_string_to_bitmap(pattern, filename)

    print("Dataset creation complete!")

if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Error: Font file {FONT_PATH} not found!")
        exit(1)

    # Create the dataset with 5000 samples for better training
    create_dataset(num_samples=5000)
