import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor
import random
import string

def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def generate_image(text, font_path, output_path, image_size):
    # Create a white background image
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # Load font; if the specified font isn't found, load the default one
    try:
        font = ImageFont.truetype(font_path, size=30)
    except IOError:
        print("Warning: Could not load specified font. Falling back to default font.")
        font = ImageFont.load_default()

    margin = 10
    max_text_width = image_size[0] - (2 * margin)
    lines = wrap_text(text, font, max_text_width, draw)

    line_spacing = 10
    total_text_height = 0
    line_heights = []
    for line in lines:
         bbox_line = draw.textbbox((0, 0), line, font=font)
         line_height = bbox_line[3] - bbox_line[1]
         line_heights.append(line_height)
         total_text_height += line_height
    total_text_height += line_spacing * (len(lines) - 1)

    y = margin
    for line, line_height in zip(lines, line_heights):
         bbox_line = draw.textbbox((0, 0), line, font=font)
         line_width = bbox_line[2] - bbox_line[0]
         x = margin
         draw.text((x, y), line, fill="black", font=font)
         y += line_height + line_spacing

    img.save(output_path)
    print(f"Saved image: {output_path}")

if __name__ == "__main__":
    # Fixed image dimensions (vertical rectangle)
    image_width = 400
    image_height = 800
    image_size = (image_width, image_height)

    font_path = "OpenSans-Regular.ttf"  # Change this to your desired TTF font file

    # Import texts from the external data.py file
    from data import texts  # Newly added import

    output_dir = "train_data"
    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
         futures = []
         # Generate images for all texts (both predefined and random).
         for i, text in enumerate(texts, start=1):
             output_path = os.path.join(output_dir, f"image_{i:02d}.png")
             futures.append(executor.submit(generate_image, text, font_path, output_path, image_size))

         for future in futures:
             # Wait for each submitted job to complete, re-raising exceptions if any.
             future.result()
