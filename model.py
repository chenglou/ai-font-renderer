"""
This file implements an attention-based neural network for rendering ASCII text as bitmap font images.

The model uses self-attention mechanisms that allow characters to influence each other's rendering,
creating a more coherent font appearance across a string. It includes a complete pipeline for:
- Training a neural font renderer on ASCII characters
- Rendering text strings as bitmap images (both console output and BMP files)
- Saving and loading trained models

Usage:
  - Run with --train flag to train a new model: python model.py --train
  - Run without arguments to load a saved model and render sample strings: python model.py

The model renders characters as 8x6 bitmaps and supports sequences up to 25 characters long.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image
import chars  # using the custom font from ascii/chars.py

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Sequence model using attention for string rendering
class AttentionFontRenderer(nn.Module):
    def __init__(self, max_length=20):
        super().__init__()
        self.max_length = max_length

        # Character embedding
        self.embedding = nn.Embedding(128, 64)  # ASCII codes to embeddings

        # Initialize positional encoding with fixed random seed
        generator = torch.Generator().manual_seed(SEED)
        self.positional_encoding = nn.Parameter(torch.randn(max_length, 64, generator=generator))

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # Processing after attention
        self.fc1 = nn.Linear(64, 96)
        self.fc2 = nn.Linear(96, 48)  # 48 = 8×6 bitmap
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len], containing ASCII codes
        batch_size, seq_len = x.shape

        # Clamp sequence length to max_length
        seq_len = min(seq_len, self.max_length)
        x = x[:, :seq_len]

        # Embed the input characters
        embedded = self.embedding(x)  # [batch_size, seq_len, 64]

        # Add positional encoding
        positions = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embedded = embedded + positions

        # Self-attention (transpose for nn.MultiheadAttention)
        attn_input = embedded.transpose(0, 1)  # [seq_len, batch_size, 64]
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, 64]

        # Generate bitmaps for each character
        x = self.activation(self.fc1(attn_output))
        bitmaps = self.output_activation(self.fc2(x))  # [batch_size, seq_len, 48]

        return bitmaps

# Generate random strings from the available characters
def generate_random_string(length):
    available_chars = list(chars.chars.keys())
    return ''.join(random.choice(available_chars) for _ in range(length))

# Create a dataset of strings
def create_string_dataset(num_samples=1000, min_length=3, max_length=15):
    inputs = []
    targets = []

    # Reset random seed for reproducible dataset generation
    random.seed(SEED)

    for _ in range(num_samples):
        # Generate a random string
        length = random.randint(min_length, max_length)
        string = generate_random_string(length)

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        inputs.append(ascii_codes)

        # Get the bitmap for each character
        string_bitmaps = []
        for c in string:
            string_bitmaps.append(chars.chars[c])
        targets.append(string_bitmaps)

    # Pad sequences to max_length
    max_len = max(len(s) for s in inputs)
    padded_inputs = []
    padded_targets = []

    for codes, bitmaps in zip(inputs, targets):
        # Pad inputs with zeros
        padded_input = codes + [0] * (max_len - len(codes))
        padded_inputs.append(padded_input)

        # Pad targets with zero bitmaps
        zero_bitmap = [0] * 48
        padded_target = bitmaps + [zero_bitmap] * (max_len - len(bitmaps))
        padded_targets.append(padded_target)

    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(padded_targets, dtype=torch.float32)

    return data.TensorDataset(inputs_tensor, targets_tensor)

# Training function for the attention model
def train_attention_model(model, dataset, num_epochs=500, lr=0.001, batch_size=32):
    # Create dataloader with fixed random seed for worker initialization
    g = torch.Generator()
    g.manual_seed(SEED)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=g, worker_init_fn=lambda id: random.seed(SEED + id))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            # Calculate loss
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        if avg_loss < 1e-4:
            print(f"Converged at epoch {epoch}, Loss: {avg_loss:.6f}")
            break
    return model

# Helper function to print a sequence of bitmaps side by side
def print_string_bitmap(bitmaps, spacing=1):
    # bitmaps shape: [seq_len, 48]
    if isinstance(bitmaps, torch.Tensor):
        bitmaps = bitmaps.detach().cpu().numpy()

    seq_len = len(bitmaps)
    char_width = 6
    char_height = 8

    # Print each row
    for row in range(char_height):
        line = ""
        for i in range(seq_len):
            # Get the corresponding row for the current character
            char_row = bitmaps[i][row * char_width:(row + 1) * char_width]
            # Convert to string representation
            char_line = ''.join(['#' if p >= 0.5 else ' ' for p in char_row])
            line += char_line + ' ' * spacing
        print(line)

# Function to save rendered strings as BMP images
def save_rendered_strings_to_bmp(model, strings):
    # Fixed parameters
    output_dir = "train_test"
    scale = 4

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for idx, string in enumerate(strings):
        # Cap string length to model's max_length
        if len(string) > model.max_length:
            string = string[:model.max_length]
            print(f"Warning: String truncated to {model.max_length} characters: {string}")

        print(f"Rendering: {string}")

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        # Pad if necessary
        if len(ascii_codes) < model.max_length:
            ascii_codes = ascii_codes + [0] * (model.max_length - len(ascii_codes))

        # Get model prediction
        x = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).squeeze(0)  # shape [seq_len, 48]
        pred = pred[:len(string)]  # Only get the actual characters

        # Convert prediction to numpy array
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        # Determine dimensions of the output image
        seq_len = len(string)
        char_width = 6
        char_height = 8
        spacing = 1  # Horizontal spacing between characters

        # Create a new image (with white background)
        img_width = (char_width * seq_len) + (spacing * (seq_len - 1))
        img_height = char_height
        img = np.ones((img_height, img_width), dtype=np.uint8) * 255

        # Fill in the image with character bitmaps
        for char_idx in range(seq_len):
            # Calculate x-position with spacing
            x_offset = char_idx * (char_width + spacing)

            # Get the bitmap for this character
            bitmap = pred[char_idx]

            # Place the bitmap in the image
            for row in range(char_height):
                for col in range(char_width):
                    pixel_value = bitmap[row * char_width + col]
                    # Set pixel black (0) if the value is >= 0.5; white otherwise
                    if pixel_value >= 0.5:
                        img[row, x_offset + col] = 0

        # Scale up the image if needed
        if scale > 1:
            img_scaled = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        else:
            img_scaled = img

        # Convert to PIL Image and save
        pil_img = Image.fromarray(img_scaled)
        filename = f"{output_dir}/string_{idx}.bmp"
        pil_img.save(filename, "BMP")

    print(f"Saved {len(strings)} rendered strings to {output_dir}/")

# Render strings to both console and image files
def render_strings(model, strings):
    """Render a list of strings to console and BMP files"""
    # Preview all strings on console
    for s in strings:
        # Cap string length to model's max_length
        if len(s) > model.max_length:
            s_capped = s[:model.max_length]
            print(f"\nString: {s} (truncated to: {s_capped})")
        else:
            print(f"\nString: {s}")
            s_capped = s

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in s_capped]
        # Pad if necessary
        if len(ascii_codes) < model.max_length:
            ascii_codes = ascii_codes + [0] * (model.max_length - len(ascii_codes))

        # Get prediction
        x = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0)
        pred = model(x).squeeze(0)  # shape [seq_len, 48]
        pred = pred[:len(s_capped)]  # Only get the actual characters

        # Print to console
        print_string_bitmap(pred)

    # Save to image files
    save_rendered_strings_to_bmp(model, strings)

# Train the string renderer
def train_string_renderer():
    print("Creating string dataset...")
    # Larger dataset with longer strings for better learning
    dataset = create_string_dataset(num_samples=2000, min_length=5, max_length=20)

    print("Training attention-based string renderer...")
    # Use a larger max_length to handle longer strings
    model = AttentionFontRenderer(max_length=25)
    model = train_attention_model(model, dataset, num_epochs=800)

    return model

def save_model(model, filename="font_renderer.pth"):
    """Save model weights to a file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename="font_renderer.pth"):
    """Load model weights from a file"""
    # Use the same max_length as in training
    model = AttentionFontRenderer(max_length=25)
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filename}")
    return model

if __name__ == '__main__':
    import sys

    # Check if --train flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Train mode: train a new model and save it
        model = train_string_renderer()
        save_model(model)
    else:
        # Render mode: load model if available, otherwise train first
        if os.path.exists("font_renderer.pth"):
            model = load_model()
        else:
            print("No saved model found. Training a new model...")
            model = train_string_renderer()
            save_model(model)

    # Render strings - edit this list to change what gets rendered
    render_strings(model, [
        "HELLO WORLD",
        "THE QUICK BROWN FOX",
        "ABCDEFGHIJKLMNOPQRST",
        "CLAUDE CODE",
        "FONT RENDERER"
    ])
