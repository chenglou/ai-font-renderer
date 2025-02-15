import chars
from PIL import Image
import os
import random
import concurrent.futures

# Generate a collage of randomly selected ASCII characters (6 columns and 3 rows)
# using a deterministic random seed. The resulting collage is saved to a BMP file.
def generate_random_collage(filename, seed):
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

if __name__ == "__main__":
    os.makedirs("train_ascii", exist_ok=True)

    # Use a ProcessPoolExecutor to parallelize the generation of 400 images.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for indices 0 to 399
        futures = [
            executor.submit(generate_random_collage, filename=f"train_ascii/image_{index}.bmp", seed=index)
            for index in range(400)
        ]
        # Wait for all tasks to complete.
        concurrent.futures.wait(futures)

    print("Finished generating 400 images.")
