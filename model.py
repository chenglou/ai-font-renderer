"""
This file implements an attention-based neural network for rendering ASCII text as bitmap font images.

The model uses self-attention mechanisms that allow characters to influence each other's rendering,
creating a coherent font appearance across a string. It includes a complete pipeline for:
- Training a neural font renderer on ASCII characters
- Rendering text strings as bitmap images
- Saving and loading trained models

Usage:
  - Run with --train flag to train a new model: python model.py --train
  - Run with --generate-samples flag to create sample training data: python model.py --generate-samples
  - Run without arguments to load a saved model and render sample strings: python model.py

Architecture learnings:
  - Single attention layer performs nearly as well as multiple layers for this task
  - A single fully connected layer after attention is sufficient (removing additional FC layers showed no quality loss)
  - Larger datasets (5000+ samples) produce significantly better quality
  - PixelShuffle upsampling significantly outperforms U-Net for font rendering, especially with repeating characters
  - Focal loss works better than standard BCE loss for this task (confirmed)
  - Early stopping based on validation helps prevent overfitting (confirmed)
  - Smaller embedding dimensions (32) work just as well as larger ones (80) while reducing memory usage (~60% reduction)
  - Learned positional encodings are CRUCIAL for this task - fixed sinusoidal encodings failed completely (99% white output), disregarding embedding dimensions (32 or 80). Maybe learned pos captures specific spatial relationships?
  - Both validation and regularization are important for generalization
  - Simpler architectures should be preferred when they perform comparably
  - Dropout regularization is needed for improved generalization
  - (FP16) was tested and produced same quality outputs as FP32, but added implementation complexity

Conv2d upsampling architecture experiments:
  - U-Net with skip connections: (val_loss: 0.011677 @ epoch 90)
    - Ok for general quality but shows pathological behavior with repeating characters (I's and W's)
    - Bottleneck layer loses character distinctiveness
  - 2-steps 4× upsampling: (val_loss: 0.009904 @ epoch 93)
  - Single-step 4× upsampling: (val_loss: 0.011142 @ epoch 95). Worse than 2-steps upsampling.
  - PixelShuffle: best results (val_loss: 0.007169 @ epoch 41). Faster training convergence.

Challenging patterns requiring special attention:
  - Sequences of repeating characters (e.g., "IIIIIIIIIIII" or "WWWWWWWWWWWW")
  - Alternating character patterns (e.g., "IWIWIWIWIWI")
  - Groups of similar characters with spaces (e.g., "IIIII IIIII IIIII")

Performance optimizations:
  - Hardware acceleration with MPS (Metal Performance Shaders) gives ~60% speedup on M-series Macs
  - Larger batch sizes (128) improve training efficiency and output quality
  - Default learning rate (0.001) with batch size 128 provides best speed/quality balance
  - Training time ~4-5 minutes on M2 Pro with these optimizations (vs 26+ minutes without)
  - Reduced model complexity (removing FC layers) further improves training efficiency

These observations are based on experimentation with this specific task and dataset.
Different font styles or character sets might require different approaches.

The model supports sheet-based rendering with a fixed output size (40x120 pixels by default),
with 4x upsampling to produce high-resolution 160x480 output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random
import os
import numpy as np
import math
from PIL import Image
import chars  # using the custom font from ascii/chars.py

# Global constants for sheet dimensions
CHAR_HEIGHT = 8
CHAR_WIDTH = 6
SHEET_HEIGHT = 40  # 5 rows of characters
SHEET_WIDTH = 120  # 20 characters per row
# Calculate how many characters can fit on a sheet
CHARS_PER_ROW = SHEET_WIDTH // CHAR_WIDTH
MAX_ROWS = SHEET_HEIGHT // CHAR_HEIGHT
MAX_CHARS_PER_SHEET = CHARS_PER_ROW * MAX_ROWS
# Default upsampling factor for high-resolution output
DEFAULT_SCALE_FACTOR = 4
# Output directory for rendered test strings
OUTPUT_DIR = "train_test_pixelshuffle_enhanced"

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure CUDA device - restrict to GPU 3 only
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Device selection logic
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

print(f"Device: {device}")

class AttentionFontRenderer(nn.Module):
    def __init__(self, max_length=MAX_CHARS_PER_SHEET, sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH, scale_factor=DEFAULT_SCALE_FACTOR):
        super().__init__()
        self.max_length = max_length
        # Keep original dimensions for initial output
        self.sheet_height = sheet_height
        self.sheet_width = sheet_width
        self.base_sheet_size = sheet_height * sheet_width
        # Store scale factor for later use
        self.scale_factor = scale_factor
        # Final output dimensions after upsampling
        self.output_height = sheet_height * scale_factor
        self.output_width = sheet_width * scale_factor

        # Reduced embedding dimension (80 → 32)
        self.embedding_dim = 32  # Reduced from 80
        self.embedding = nn.Embedding(128, self.embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)

        # Learned positional encoding with reduced dimension
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, self.embedding_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Single attention layer with reduced dimension
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, dropout=0.1)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        # Processing network (simplified) - adjusted dimensions
        self.fc1 = nn.Linear(self.embedding_dim, 64)  # Reduced from 160
        self.dropout1 = nn.Dropout(0.15)

        # Generate base resolution bitmap first
        self.fc_base = nn.Linear(64 * max_length, self.base_sheet_size)

        # PixelShuffle approach for efficient upsampling
        # Uses sub-pixel convolution (pixel shuffle) which tends to learn sharper details
        self.upsample = nn.Sequential(
            # First extract features at base resolution
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # First pixel shuffle - 2x upscale
            # Outputs 16 channels but expands to 64 first for the pixel shuffle
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 64 -> 16 channels, 2x spatial size
            nn.ReLU(),

            # Second pixel shuffle - another 2x upscale
            # Outputs 4 channels but expands to 16 first for the pixel shuffle
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Extra processing
            nn.ReLU(),

            # Final pixelshuffle + refine
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 16 -> 4 channels, 2x spatial size
            nn.Conv2d(4, 1, kernel_size=3, padding=1),  # Final output
            nn.Sigmoid()
        )

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len], containing ASCII codes
        batch_size, seq_len = x.shape

        # Clamp sequence length to max_length
        seq_len = min(seq_len, self.max_length)
        x = x[:, :seq_len]

        # Embed the input characters with dropout
        embedded = self.embedding(x)  # [batch_size, seq_len, self.embedding_dim]
        embedded = self.embedding_dropout(embedded)

        # Add positional encoding
        positions = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embedded = embedded + positions

        # Apply single attention layer with residual connection
        attn_input = embedded.transpose(0, 1)  # [seq_len, batch_size, self.embedding_dim]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, self.embedding_dim]

        # Add residual connection and normalize
        attn_output = self.layer_norm(embedded + attn_output)

        # Process through reduced fully connected layers
        x = self.activation(self.fc1(attn_output))  # [batch_size, seq_len, 64]
        x = self.dropout1(x)

        # Reshape to connect all character features
        x = x.reshape(batch_size, -1)  # [batch_size, seq_len * 64]

        # Zero-pad if sequence is shorter than max_length
        if seq_len < self.max_length:
            padding = torch.zeros(batch_size, (self.max_length - seq_len) * 64,
                                device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Generate the base resolution bitmap
        base_sheet = self.output_activation(self.fc_base(x))  # [batch_size, base_sheet_size]

        # Reshape to proper dimensions for upsampling
        base_sheet = base_sheet.view(batch_size, 1, self.sheet_height, self.sheet_width)

        # Apply PixelShuffle upsampling layers
        sheet_with_channel = self.upsample(base_sheet)  # [batch_size, 1, output_height, output_width]

        # Remove the channel dimension for output
        sheet = sheet_with_channel.squeeze(1)  # [batch_size, output_height, output_width]

        return sheet

# Generate random strings from the available characters
def generate_random_string(length):
    available_chars = list(chars.chars.keys())
    return ''.join(random.choice(available_chars) for _ in range(length))

# Place string characters as bitmap on a target sheet
def place_string_on_sheet(string, target_sheet):
    """
    Places a string on a target sheet as bitmaps.

    Args:
        string (str): The string to render
        target_sheet (numpy.ndarray): Target array to place characters on

    Returns:
        numpy.ndarray: Updated target sheet with rendered string
    """
    char_idx = 0
    for row in range(MAX_ROWS):
        if char_idx >= len(string):
            break

        for col in range(CHARS_PER_ROW):
            if char_idx >= len(string):
                break

            # Get character bitmap
            if string[char_idx] in chars.chars:
                char_bitmap = chars.chars[string[char_idx]]

                # Calculate position in the sheet
                y_start = row * CHAR_HEIGHT
                x_start = col * CHAR_WIDTH

                # Place character bitmap in the sheet
                for y in range(CHAR_HEIGHT):
                    for x in range(CHAR_WIDTH):
                        bitmap_idx = y * CHAR_WIDTH + x
                        if bitmap_idx < len(char_bitmap):
                            target_sheet[y_start + y, x_start + x] = char_bitmap[bitmap_idx]

            char_idx += 1

    return target_sheet

# Create a dataset of text sheets and save sample images to a folder
def create_string_dataset(num_samples=1000, min_length=20, max_length=MAX_CHARS_PER_SHEET,
                         sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH, char_height=CHAR_HEIGHT, char_width=CHAR_WIDTH,
                         save_samples=False, samples_dir="train_input", num_samples_to_save=10):
    """Create a dataset of text sheets with consistent character grid dimensions."""
    # Reset random seed for reproducible dataset generation
    random.seed(SEED)

    # Use the global constants
    max_chars_per_sheet = MAX_CHARS_PER_SHEET

    # Pre-allocate arrays for better performance
    all_inputs = []
    all_strings = []  # Store generated strings for reference
    all_targets = np.zeros((num_samples, sheet_height, sheet_width), dtype=np.float32)

    # Create output directory if saving samples
    if save_samples:
        os.makedirs(samples_dir, exist_ok=True)
        print(f"Saving {min(num_samples, num_samples_to_save)} sample sheets to {samples_dir}/")

    for sample_idx in range(num_samples):
        # Generate a random string that fits in the sheet
        length = random.randint(min_length, min(max_length, max_chars_per_sheet))
        string = generate_random_string(length)
        all_strings.append(string)

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        all_inputs.append(ascii_codes)

        # Place string on sheet using shared function
        place_string_on_sheet(string, all_targets[sample_idx])

        # Save this sample as an image if requested
        if save_samples and sample_idx < num_samples_to_save:
            # Convert binary sheet to image
            img = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
            for y in range(sheet_height):
                for x in range(sheet_width):
                    if all_targets[sample_idx, y, x] >= 0.5:
                        img[y, x] = 0

            # Convert to PIL Image and save
            pil_img = Image.fromarray(img)
            filename = f"{samples_dir}/input_{sample_idx}_{string[:20]}.bmp"
            pil_img.save(filename, "BMP")

            # Also save the input string for reference
            with open(f"{samples_dir}/input_{sample_idx}_text.txt", "w") as f:
                f.write(string)

    # Pad sequences to max_length
    max_len = max(len(s) for s in all_inputs)
    padded_inputs = np.zeros((num_samples, max_len), dtype=np.int64)

    for i, codes in enumerate(all_inputs):
        # Pad inputs with zeros
        padded_inputs[i, :len(codes)] = codes

    # Convert to tensors
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

    if save_samples:
        print(f"Dataset creation complete: {num_samples} samples with dimensions {sheet_height}x{sheet_width}")

    return data.TensorDataset(inputs_tensor, targets_tensor)

# Balanced training function with focal loss and moderate regularization
def train_attention_model(model, dataset, num_epochs=500, lr=0.001, batch_size=32,
                         early_stopping_patience=15, validation_split=0.1):
    # Create additional validation samples with specific patterns
    # These will be added to the validation set to ensure model handles them well
    additional_val_samples = 20  # Add 20 validation samples with specific patterns

    # Create inputs with repeating patterns
    pattern_inputs = []
    pattern_targets = []

    # Generate the patterns
    patterns = [
        "IIIIIIIIIIIIIIIIIIII",  # Repeating I's
        "WWWWWWWWWWWWWWWWWWWW",  # Repeating W's
        "IIIII IIIII IIIII IIIII",  # Groups of I's with spaces
        "WWWWW WWWWW WWWWW WWWWW",  # Groups of W's with spaces
        "IWIWIWIWIWIWIWIWIWIWI",  # Alternating I and W pattern
        "                     ",
    ]

    for pattern in patterns:
        for _ in range(additional_val_samples // len(patterns)):
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

    # Concatenate pattern samples
    pattern_inputs = torch.cat(pattern_inputs, dim=0)
    pattern_targets = torch.cat(pattern_targets, dim=0)
    pattern_dataset = data.TensorDataset(pattern_inputs, pattern_targets)

    # Split the original dataset into training and validation
    orig_dataset_size = len(dataset)
    val_size = int(validation_split * orig_dataset_size) - additional_val_samples  # Adjust to account for pattern samples
    train_size = orig_dataset_size - val_size

    print(f"Dataset split: {train_size} training samples, {val_size + additional_val_samples} validation samples")

    # Split the original dataset
    train_dataset, val_dataset_orig = data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Combine the original validation set with our pattern samples
    val_dataset = data.ConcatDataset([val_dataset_orig, pattern_dataset])

    # Create dataloaders with fixed random seed
    g = torch.Generator()
    g.manual_seed(SEED)

    # Optimize DataLoader for GPU usage
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda id: random.seed(SEED + id)
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
        num_workers=2,
        pin_memory=True
    )

    # Focal loss - known to work well for this task
    def focal_bce_loss(pred, target, gamma=2.0, alpha=0.25):
        bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')

        # Calculate focal weights - focus on hard examples
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** gamma

        # Alpha weighting for positive pixels (text)
        alpha_weight = torch.where(target > 0.5, alpha, 1 - alpha)

        # Combine weights and take mean
        loss = focal_weight * alpha_weight * bce_loss
        return loss.mean()

    # AdamW optimizer with moderate weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Simple learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            # Move inputs and targets to device with non-blocking transfers
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(batch_inputs)

            # Model is working correctly, no need for debug shapes anymore

            # Since we're training with 4x upsampling, the model outputs will be 4x larger
            # We need to handle this by upsampling the targets
            if hasattr(model, 'output_height') and hasattr(model, 'output_width'):
                # Create a tensor to hold the upsampled targets
                batch_size = batch_targets.shape[0]
                upsampled_targets = torch.nn.functional.interpolate(
                    batch_targets.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
                    size=(model.output_height, model.output_width),
                    mode='nearest'
                ).squeeze(1)  # Remove channel dimension [B, H, W]
                batch_targets = upsampled_targets
            else:
                # Standard case - just reshape
                batch_targets = batch_targets.view(outputs.shape)

            loss = focal_bce_loss(outputs, batch_targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                # Move inputs and targets to device with non-blocking transfers
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)

                # Forward pass
                outputs = model(batch_inputs)

                # Since we're training with 4x upsampling, the model outputs will be 4x larger
                # We need to handle this by upsampling the targets
                if hasattr(model, 'output_height') and hasattr(model, 'output_width'):
                    # Create a tensor to hold the upsampled targets
                    batch_size = batch_targets.shape[0]
                    upsampled_targets = torch.nn.functional.interpolate(
                        batch_targets.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
                        size=(model.output_height, model.output_width),
                        mode='nearest'
                    ).squeeze(1)  # Remove channel dimension [B, H, W]
                    batch_targets = upsampled_targets
                else:
                    # Standard case - just reshape
                    batch_targets = batch_targets.view(outputs.shape)

                val_loss = focal_bce_loss(outputs, batch_targets)
                total_val_loss += val_loss.item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch}, New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, Best Val Loss: {best_val_loss:.6f}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

    # Ensure best model is loaded
    if best_model_state is not None and patience_counter < early_stopping_patience:
        model.load_state_dict(best_model_state)
        print(f"Training completed, Best Val Loss: {best_val_loss:.6f}")

    return model

# Function to save rendered sheets as BMP images
def render_strings(model, strings, output_dir):
    """Render a list of strings as BMP images"""

    for idx, string in enumerate(strings):
        # Cap string length to model's max_length
        if len(string) > model.max_length:
            string = string[:model.max_length]
            print(f"Warning: String truncated to {model.max_length} characters: {string}")

        # Convert to ASCII codes
        ascii_codes = [ord(c) for c in string]
        # Pad if necessary
        if len(ascii_codes) < model.max_length:
            ascii_codes = ascii_codes + [0] * (model.max_length - len(ascii_codes))

        # Get model prediction
        x = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            sheet = model(x).squeeze(0)  # shape will be [output_height, output_width]

        # Convert prediction to numpy array
        if isinstance(sheet, torch.Tensor):
            sheet = sheet.detach().cpu().numpy()

        # Get dimensions - handle both upsampled and regular outputs
        sheet_height = sheet.shape[0]
        sheet_width = sheet.shape[1]

        # Create a new image (with white background)
        img = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255

        # Fill in the image with the predicted bitmap
        for row in range(sheet_height):
            for col in range(sheet_width):
                # Set pixel black (0) if the value is >= 0.5; white otherwise
                if sheet[row, col] >= 0.5:
                    img[row, col] = 0

        # Convert to PIL Image and save
        pil_img = Image.fromarray(img)
        filename = f"{output_dir}/string_{idx}.bmp"
        pil_img.save(filename, "BMP")

    print(f"Saved {len(strings)} rendered strings to {output_dir}/")

# Train the sheet-based renderer
def train_string_renderer(generate_only=False):
    print("Creating sheet dataset...")

    # Create the dataset and save samples to the train_input folder
    dataset = create_string_dataset(
        num_samples=5000,  # Increased to 5000 samples for better learning
        min_length=10,
        max_length=MAX_CHARS_PER_SHEET,
        save_samples=True,  # Save sample images
        samples_dir="train_input",
        num_samples_to_save=10  # Save 10 samples for reference
    )

    # If generate_only flag is set, return without training
    if generate_only:
        print("Dataset generation complete. Skipping training.")
        return None

    print("Training attention-based sheet renderer with reduced embedding dimensions (32) and learned positional encoding...")
    # Initialize model with sheet dimensions and native upscaling
    model = AttentionFontRenderer(
        max_length=MAX_CHARS_PER_SHEET,
        sheet_height=SHEET_HEIGHT,
        sheet_width=SHEET_WIDTH,
        scale_factor=DEFAULT_SCALE_FACTOR
    )

    # Move model to device
    model = model.to(device)

    # Adjust batch size based on hardware
    if torch.cuda.is_available():
        # Larger batch size for GPU
        batch_size = 256  # CUDA can handle larger batches
    else:
        # Original batch size for CPU/MPS
        batch_size = 128

    print(f"Using batch size {batch_size}")

    model = train_attention_model(
        model,
        dataset,
        num_epochs=100,
        batch_size=batch_size,
        early_stopping_patience=15,  # More patience to see learning curve
        validation_split=0.1  # 10% validation data
    )

    return model

def save_model(model, filename="font_renderer.pth"):
    """Save model weights to a file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename="font_renderer.pth"):
    """Load model weights from a file"""
    # Initialize model with global constants
    model = AttentionFontRenderer(
        max_length=MAX_CHARS_PER_SHEET,
        sheet_height=SHEET_HEIGHT,
        sheet_width=SHEET_WIDTH,
        scale_factor=DEFAULT_SCALE_FACTOR
    )
    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filename}")
    return model

if __name__ == '__main__':
    import sys

    # Ensure output directory exists at start
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_strings = [
        "HELLO LEANN I LOVE YOU SO MUCH I HOPE YOU HAVE A GREAT DAY",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "WWWWWWWWWWWWWWWWWWWW",  # Width test (repeating wide character)
        "IIIIIIIIIIIIIIIIIIII",  # Width test (repeating narrow character)
        "ALTERNATING CASE TEST   SPACES",  # Spacing test
        "CLAUDE IS RENDERING FONTS",
        "ZYXWVUTSRQPONMLKJIHGFEDCBA",  # Reverse alphabet
        "AEIOU BCDFGHJKLMNPQRSTVWXYZ",  # Vowels and consonants grouped
        "EXACTLY TWENTY CHARS",  # Boundary test
        "                    ",
    ]
    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            # Train mode: train a new model and save it
            model = train_string_renderer()
            save_model(model)

            # Render test strings
            render_strings(model, test_strings, output_dir=OUTPUT_DIR)
        elif sys.argv[1] == "--generate-samples":
            # Generate samples only without training
            train_string_renderer(generate_only=True)
            print("Sample generation complete. Check the train_input/ directory.")
            sys.exit(0)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options: --train, --generate-samples")
            sys.exit(1)
    else:
        # Render mode: load model if available, otherwise train first
        if os.path.exists("font_renderer.pth"):
            model = load_model()
        else:
            print("No saved model found. Training a new model...")
            model = train_string_renderer()
            save_model(model)

        # Render strings - edit this list to change what gets rendered
        render_strings(model, test_strings, output_dir=OUTPUT_DIR)
