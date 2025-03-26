"""
This file implements an attention-based neural network for rendering ASCII text as bitmap font images.

The model uses self-attention mechanisms that allow characters to influence each other's rendering,
creating a coherent font appearance across a string. It includes a complete pipeline for:
- Training a neural font renderer on ASCII characters
- Rendering text strings as bitmap images
- Saving and loading trained models

Usage:
  - Run with --train flag to train a new model: python model.py --train
  - Run without arguments to load a saved model and render sample strings: python model.py

Architecture learnings:
  - Single attention layer performs nearly as well as multiple layers for this task
  - A single fully connected layer after attention is sufficient (removing additional FC layers showed no quality loss)
  - Larger datasets produce significantly better quality. For old hand-crafted pixel font, loss was ~0.002 with 5k, 0.000193 with 15k, and further improved with 25k samples.
  - Indirect rendering via downsampled feature map and upsampling for better scalability
  - For grayscale output, standard MSE loss works better than focal or custom losses
  - Clamped linear output (replacing sigmoid) provides sharper gradients for training
  - Early stopping based on validation helps prevent overfitting (confirmed)
  - Smaller embedding dimensions (32) work just as well as larger ones (80) while reducing memory usage (~60% reduction)
  - Learned positional encodings are CRUCIAL for this task - fixed sinusoidal encodings failed completely (99% white output)
  - Both validation and regularization are important for generalization
  - Simpler architectures should be preferred when they perform comparably
  - Dropout regularization is needed for improved generalization
  - (FP16) was tested and produced same quality outputs as FP32, but added implementation complexity

Architecture improvements:
  - Replaced large FC layer with smaller FC + convolution + pixel shuffle upsampling
  - FC layer outputs downsampled feature map (reducing parameters by ~75%)
  - Single convolution expands features for pixel shuffle upsampling
  - Clamped linear output (instead of sigmoid) provides sharper gradients
  - Grayscale rendering preserves antialiasing in the font output
  - Architecture scales better to larger image dimensions

Challenging patterns requiring special attention:
  - Sequences of repeating characters (e.g., "IIIIIIIIIIII" or "WWWWWWWWWWWW")
  - Alternating character patterns (e.g., "IWIWIWIWIWI")
  - Groups of similar characters with spaces (e.g., "IIIII IIIII IIIII")

Performance optimizations:
  - Hardware acceleration with MPS (Metal Performance Shaders) gives ~60% speedup on M-series Macs
  - Optimal batch sizes (256 for CPU/MPS, 1024 for GPU) improve training efficiency
  - Increased learning rate (from 0.0005 to 0.004) speeds up training with the new architecture
  - Doubled dataset size (50k samples) for better generalization
  - Reduced parameter count (~75% reduction in FC layer) enables faster training
  - Memory-efficient architecture allows training on higher resolution outputs
  - Training time significantly reduced with these optimizations

These observations are based on experimentation with this specific task and dataset.
Different font styles or character sets might require different approaches.

The model supports sheet-based rendering with dimensions defined in generate_font.py
without any upsampling, producing output at native resolution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random
import os
import numpy as np
import generate_font  # Import the font generation module
from generate_font import SHEET_HEIGHT, SHEET_WIDTH
from generate_font import MAX_CHARS_PER_SHEET

# No upsampling - using original dimensions
# Output directory for rendered test strings
OUTPUT_DIR = "train_test_pixelshuffle"

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
    def __init__(self, max_length=MAX_CHARS_PER_SHEET):
        super().__init__()
        self.max_length = max_length

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

        # FC layer outputs downsampled feature map (1 channel at half resolution)
        self.fc_output = nn.Linear(64 * max_length, (SHEET_HEIGHT//2) * (SHEET_WIDTH//2))

        # Conv layer expands features for pixel shuffle (1→4 channels)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

        # Pixel shuffle for upsampling
        self.pixel_shuffle = nn.PixelShuffle(2)  # scale factor 2

        self.activation = nn.ReLU()
        # Use clamped linear output instead of sigmoid
        self.output_activation = lambda x: torch.clamp(x, 0.0, 1.0)

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

        # Generate downsampled feature map
        x = self.fc_output(x)  # [batch_size, (SHEET_HEIGHT//2) * (SHEET_WIDTH//2)]

        # Reshape to [batch_size, 1, SHEET_HEIGHT//2, SHEET_WIDTH//2]
        x = x.view(batch_size, 1, SHEET_HEIGHT//2, SHEET_WIDTH//2)

        # Apply conv to expand features (1→4 channels)
        x = self.activation(self.conv(x))  # [batch_size, 4, SHEET_HEIGHT//2, SHEET_WIDTH//2]

        # Apply pixel shuffle upsampling
        sheet = self.pixel_shuffle(x)  # [batch_size, 1, SHEET_HEIGHT, SHEET_WIDTH]

        # Apply output activation and remove channel dimension
        sheet = self.output_activation(sheet).squeeze(1)  # [batch_size, SHEET_HEIGHT, SHEET_WIDTH]

        return sheet

# Use the generate_font module to create datasets

# Balanced training function with focal loss and moderate regularization
def train_attention_model(model, dataset, batch_size):
    num_epochs=200
    lr=0.004
    early_stopping_patience=15
    validation_split=0.1
    # Create additional validation samples with specific patterns
    # Get challenging pattern dataset from generate_font
    pattern_dataset = generate_font.generate_challenging_patterns()
    additional_val_samples = len(pattern_dataset)  # Get the number of additional samples

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

    # Plain MSE loss for grayscale values
    def focal_mse_loss(pred, target, contrast_factor=None):
        # Using pure MSE loss for grayscale
        # Keeping function name for compatibility
        return nn.functional.mse_loss(pred, target)

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

            # Ensure targets and outputs have matching shapes
            batch_targets = batch_targets.view(outputs.shape)

            loss = focal_mse_loss(outputs, batch_targets)

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

                # Ensure targets and outputs have matching shapes
                batch_targets = batch_targets.view(outputs.shape)

                val_loss = focal_mse_loss(outputs, batch_targets)
                total_val_loss += val_loss.item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)

        # Early stopping check
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if epoch % 10 == 0:
            status = f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            if is_best:
                status += f" (New Best)"
            print(status)
        elif is_best:
            print(f"Epoch {epoch}, New best validation loss: {avg_val_loss:.6f}")

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
    os.makedirs(output_dir, exist_ok=True)

    for idx, string in enumerate(strings):
        # Cap string length to model's max_length
        if len(string) > model.max_length:
            string = string[:model.max_length]
            print(f"Warning: String truncated to {model.max_length} characters: {string}")

        # Convert to ASCII codes and pad
        ascii_codes = [ord(c) for c in string]
        if len(ascii_codes) < model.max_length:
            ascii_codes = ascii_codes + [0] * (model.max_length - len(ascii_codes))

        # Get model prediction
        x = torch.tensor(ascii_codes, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            sheet = model(x).squeeze(0)  # shape will be [SHEET_HEIGHT, SHEET_WIDTH]

        # Convert to numpy if needed
        if isinstance(sheet, torch.Tensor):
            sheet = sheet.detach().cpu().numpy()

        # Convert binary array to image and save
        filename = f"{output_dir}/string_{idx}.bmp"
        generate_font.binary_array_to_image(sheet, output_path=filename)

    print(f"Saved {len(strings)} rendered strings to {output_dir}/")

# Train the sheet-based renderer
def train_string_renderer():
    print("Creating sheet dataset...")

    # Create the dataset and save samples to the train_input folder
    dataset = generate_font.create_string_dataset(
        num_samples=50000,
        min_length=10,
        samples_dir="train_input",
        num_samples_to_save=10  # Save 10 samples for reference
    )

    print("Training attention-based sheet renderer with reduced embedding dimensions (32) and learned positional encoding...")
    # Initialize model
    model = AttentionFontRenderer(max_length=MAX_CHARS_PER_SHEET)

    # Move model to device
    model = model.to(device)

    # Adjust batch size based on hardware
    if torch.cuda.is_available():
        batch_size = 1024
    else:
        batch_size = 256

    print(f"Using batch size {batch_size}")

    model = train_attention_model(
        model,
        dataset,
        batch_size,
    )

    return model

def save_model(model, filename="font_renderer.pth"):
    """Save model weights to a file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename="font_renderer.pth"):
    """Load model weights from a file"""
    # Initialize model with defaults from imports
    model = AttentionFontRenderer(
        max_length=MAX_CHARS_PER_SHEET
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

    # Test strings for model evaluation
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
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options: --train")
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
