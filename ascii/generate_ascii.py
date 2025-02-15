import chars
from PIL import Image
import os
import random   # <-- Added to support deterministic random behavior

def visualizeInTerminalForDebug():
    for char, grid in chars.chars.items():
        print(f"Character: '{char}'")
        # Each character is defined on an 8x6 grid.
        for row in range(8):
            # Slice out each row's data from our grid data.
            row_data = grid[row * 6:(row + 1) * 6]
            # If the cell is 1, print a black cell; if 0, print a white cell.
            line = ''.join("█" if pixel == 1 else " " for pixel in row_data)
            print(line)
        print()  # Add an empty line between characters

# This function uses a deterministic random (by setting a fixed seed)
# to select 18 characters (6 per row, 3 rows). It combines their 8x6 grids
# into one single collage image and saves it as a BMP file.
def generate_random_collage(filename="ascii_collage.bmp", seed=42):
    random.seed(seed)
    ascii_keys = list(chars.chars.keys())
    # Pick 18 characters (6 columns x 3 rows)
    collage_chars = [random.choice(ascii_keys) for _ in range(18)]

    # Each character image is 6 pixels wide and 8 pixels tall.
    collage_width = 6 * 6   # 6 characters wide (6*6 = 36 pixels)
    collage_height = 3 * 8  # 3 characters tall (3*8 = 24 pixels)
    collage = Image.new("L", (collage_width, collage_height), color=255)  # white background

    for idx, char in enumerate(collage_chars):
        col = idx % 6
        row = idx // 6
        x_offset = col * 6
        y_offset = row * 8
        grid = chars.chars[char]
        for r in range(8):
            for c in range(6):
                # Set pixel black (0) if the value is 1; white otherwise.
                if grid[r * 6 + c] == 1:
                    collage.putpixel((x_offset + c, y_offset + r), 0)
    collage.save(filename, "BMP")
    print(f"Saved collage to {filename}")

# draws a collage of 6x3 randomly (but deterministically) selected characters.
def visualizeForDebug():
    """
    Generate a collage of randomly selected ASCII characters (6 columns and 3 rows)
    using a deterministic random seed. The resulting collage is saved to a BMP file.
    """
    generate_random_collage(filename="ascii_collage.bmp")

if __name__ == "__main__":
    # visualizeInTerminalForDebug()
    # The refactored version now creates one collage image.
    visualizeForDebug()
