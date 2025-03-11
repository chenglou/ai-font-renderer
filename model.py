"""
This file implements an attention-based neural network for rendering text as bitmap font images.

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
  - Larger datasets (5000+ samples) produce significantly better quality
  - PixelShuffle upsampling significantly outperforms U-Net for font rendering, especially with repeating characters
  - Focal loss works better than standard BCE loss for this task (confirmed)
  - Early stopping based on validation helps prevent overfitting (confirmed)
  - A balanced model size with moderate embedding dimensions (80) works well
  - Both validation and regularization are important for generalization
  - Simpler architectures should be preferred when they perform comparably

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

# WIP: High-Resolution Font Rendering Architectural Considerations

We're exploring expanding the model to handle higher-resolution monospace and proportional fonts. 
The current approach with a large FC layer becomes memory-intensive with high-resolution inputs.
Two promising alternatives under consideration:

1. Fully Convolutional Approach with Attention:
   - Replace the large FC layer with convolutional layers
   - Maintain attention mechanism but operate in 2D space
   - Memory-efficient for high-resolution inputs
   - Well-suited for both monospace and proportional fonts
   - Works well when font rendering requires character-to-character awareness
   - Less prone to out-of-memory issues on large inputs

2. Vision Transformer (ViT) Inspired Approach:
   - Split the input image into patches and process with a transformer
   - Similar to ViT but specialized for font rendering
   - Maintains character relationships through self-attention
   - Could better handle spatial relationships in proportional fonts
   - May require larger datasets to train effectively

The scale_factor parameter remains useful for:
- Training on lower-resolution and upscaling for specific use cases
- Managing memory-performance tradeoffs
- Potential super-resolution applications

Current work focuses on loading pre-generated high-resolution monospace font bitmaps
from generate_font_bitmap.py as the first step toward these architectural improvements.
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

# Import our monospace font dimensions from generate_font_bitmap.py
import importlib.util
spec = importlib.util.spec_from_file_location("generate_font_bitmap", "generate_font_bitmap.py")
font_bitmap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(font_bitmap_module)

# Use constants from our font bitmap generator
CHARS_PER_ROW = font_bitmap_module.CHARS_PER_ROW  # 20 chars per row
MAX_ROWS = font_bitmap_module.NUM_ROWS  # 5 rows
SHEET_WIDTH = font_bitmap_module.SHEET_WIDTH  # Based on FiraCode measurements
SHEET_HEIGHT = font_bitmap_module.SHEET_HEIGHT  # Based on FiraCode measurements
CHAR_WIDTH = SHEET_WIDTH // CHARS_PER_ROW  # Derived from sheet width
CHAR_HEIGHT = SHEET_HEIGHT // MAX_ROWS  # Derived from sheet height
MAX_CHARS_PER_SHEET = CHARS_PER_ROW * MAX_ROWS  # Total chars per sheet

# No upsampling needed for high-resolution inputs
DEFAULT_SCALE_FACTOR = 1  # Disable upsampling since input resolution is already good
# Output directory for rendered test strings
OUTPUT_DIR = "train_test_monospace_font"

print(f"Sheet configuration from generate_font_bitmap.py:")
print(f"- Character dimensions: {CHAR_WIDTH}x{CHAR_HEIGHT} pixels")
print(f"- Sheet dimensions: {SHEET_WIDTH}x{SHEET_HEIGHT} pixels")
print(f"- Characters per sheet: {MAX_CHARS_PER_SHEET}")
print(f"- Upsampling factor: {DEFAULT_SCALE_FACTOR}x")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure CUDA devices - restrict to GPUs 3, 4, 5 as requested
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

# Device selection logic
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

# Modified model with single attention layer to test its importance
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

        # Keep the same embedding size
        self.embedding = nn.Embedding(128, 80)
        self.embedding_dropout = nn.Dropout(0.1)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, 80))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Single attention layer
        self.attention = nn.MultiheadAttention(embed_dim=80, num_heads=4, dropout=0.1)
        self.layer_norm = nn.LayerNorm(80)

        # Processing network (simplified)
        self.fc1 = nn.Linear(80, 160)
        self.dropout1 = nn.Dropout(0.15)

        # Generate base resolution bitmap first
        self.fc_base = nn.Linear(160 * max_length, self.base_sheet_size)

        # For scale_factor=1, we just do some refinement without changing dimensions
        if self.scale_factor == 1:
            self.upsample = nn.Sequential(
                # Extract features at base resolution
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                
                # Intermediate processing
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                
                # Extra detail refinement
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                
                # Final output layer
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            # Original PixelShuffle approach for efficient upsampling (if scale_factor > 1)
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
        embedded = self.embedding(x)  # [batch_size, seq_len, 80]
        embedded = self.embedding_dropout(embedded)

        # Add positional encoding
        positions = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embedded = embedded + positions

        # Apply single attention layer with residual connection
        attn_input = embedded.transpose(0, 1)  # [seq_len, batch_size, 80]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, 80]

        # Add residual connection and normalize
        attn_output = self.layer_norm(embedded + attn_output)

        # Process through reduced fully connected layers
        x = self.activation(self.fc1(attn_output))  # [batch_size, seq_len, 160]
        x = self.dropout1(x)

        # Reshape to connect all character features
        x = x.reshape(batch_size, -1)  # [batch_size, seq_len * 160]

        # Zero-pad if sequence is shorter than max_length
        if seq_len < self.max_length:
            padding = torch.zeros(batch_size, (self.max_length - seq_len) * 160,
                                device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Generate the base resolution bitmap
        base_sheet = self.output_activation(self.fc_base(x))  # [batch_size, base_sheet_size]

        # Reshape to proper dimensions for processing
        base_sheet = base_sheet.view(batch_size, 1, self.sheet_height, self.sheet_width)
        
        # Apply processing layers (with or without upsampling)
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

# Create a dataset from pre-generated monospace font bitmap images
def create_string_dataset(num_samples=5000, samples_dir="train_monospace_input"):
    """Load pre-generated monospace font images as training data."""
    # Reset random seed for reproducible dataset generation
    random.seed(SEED)
    
    # Check if the directory exists
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Dataset directory '{samples_dir}' not found. Run generate_font_bitmap.py first.")
    
    # Get list of all bitmap files in the directory - our naming convention is {idx}_{text}.bmp
    bitmap_files = [f for f in os.listdir(samples_dir) 
                   if f.endswith('.bmp') and not f.startswith('special_')]
    
    if len(bitmap_files) == 0:
        raise ValueError(f"No bitmap files found in {samples_dir}. Run generate_font_bitmap.py first.")
        
    print(f"Found {len(bitmap_files)} bitmap files in {samples_dir}")
    
    # Limit to num_samples if needed
    if len(bitmap_files) > num_samples:
        bitmap_files = bitmap_files[:num_samples]
    
    # Pre-allocate arrays
    all_inputs = []
    all_targets = np.zeros((len(bitmap_files), SHEET_HEIGHT, SHEET_WIDTH), dtype=np.float32)
    
    for idx, bitmap_file in enumerate(bitmap_files):
        # Extract the ID and partial string from the filename
        # Format: X_STRING.bmp (where X is a number)
        file_parts = bitmap_file.split('_', 1)
        if len(file_parts) >= 2:
            # Extract the partial string from the filename
            partial_string = file_parts[1].replace('.bmp', '')
            
            # Convert the partial string back to a full string
            string = partial_string.replace('_', ' ')
            string = string[:MAX_CHARS_PER_SHEET]
            
            # Convert to ASCII codes
            ascii_codes = [ord(c) for c in string]
            all_inputs.append(ascii_codes)
            
            # Load the bitmap image
            bitmap_path = os.path.join(samples_dir, bitmap_file)
            img = Image.open(bitmap_path).convert('L')  # Convert to grayscale
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Ensure the image dimensions match what we expect
            if img_array.shape[0] != SHEET_HEIGHT or img_array.shape[1] != SHEET_WIDTH:
                print(f"Warning: Image file {bitmap_file} has dimensions {img_array.shape}, resizing to {SHEET_HEIGHT}x{SHEET_WIDTH}")
                # Resize the image to match expected dimensions
                img_resized = Image.fromarray(img_array).resize((SHEET_WIDTH, SHEET_HEIGHT), Image.LANCZOS)
                img_array = np.array(img_resized)
            
            # Normalize to 0-1 range and invert colors (black text on white background)
            all_targets[idx] = (255 - img_array) / 255.0
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Loaded {idx + 1}/{len(bitmap_files)} images")
    
    # Pad sequences to max_length
    max_len = max(len(codes) for codes in all_inputs)
    padded_inputs = np.zeros((len(bitmap_files), max_len), dtype=np.int64)
    
    for i, codes in enumerate(all_inputs):
        # Pad inputs with zeros
        padded_inputs[i, :len(codes)] = codes
    
    # Convert to tensors
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
    
    print(f"Dataset loading complete: {len(bitmap_files)} samples with dimensions {SHEET_HEIGHT}x{SHEET_WIDTH}")
    
    return data.TensorDataset(inputs_tensor, targets_tensor)

# Balanced training function with focal loss and moderate regularization
def train_attention_model(model, dataset, num_epochs=500, lr=0.001, batch_size=32,
                         early_stopping_patience=15, validation_split=0.1):
    # Split the original dataset into training and validation
    orig_dataset_size = len(dataset)
    
    # Calculate validation size, ensuring it's at least 1% of the dataset
    val_size = max(int(validation_split * orig_dataset_size), int(0.01 * orig_dataset_size))
    # Ensure we have enough training samples
    train_size = orig_dataset_size - val_size
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Split the original dataset
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Create dataloaders with fixed random seed
    g = torch.Generator()
    g.manual_seed(SEED)
    
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
        train_steps = 0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            # Move inputs and targets to device
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(batch_inputs)

            # Ensure targets match the model output dimensions
            if outputs.shape != batch_targets.shape:
                # Use the output shape to determine how to resize targets
                upsampled_targets = torch.nn.functional.interpolate(
                    batch_targets.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
                    size=(outputs.shape[1], outputs.shape[2]),  # Use exact output dimensions
                    mode='nearest'
                ).squeeze(1)  # Remove channel dimension [B, H, W]
                batch_targets = upsampled_targets

            loss = focal_bce_loss(outputs, batch_targets)

            # Standard backward pass
            loss.backward()
            optimizer.step()
                
            total_train_loss += loss.item()
            train_steps += 1

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                # Move inputs and targets to device
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)

                # Forward pass
                outputs = model(batch_inputs)

                # Ensure targets match the model output dimensions
                if outputs.shape != batch_targets.shape:
                    # Use the output shape to determine how to resize targets
                    upsampled_targets = torch.nn.functional.interpolate(
                        batch_targets.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
                        size=(outputs.shape[1], outputs.shape[2]),  # Use exact output dimensions
                        mode='nearest'
                    ).squeeze(1)  # Remove channel dimension [B, H, W]
                    batch_targets = upsampled_targets

                val_loss = focal_bce_loss(outputs, batch_targets)
                total_val_loss += val_loss.item()
                val_steps += 1

        # Calculate average losses
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        # Update learning rate based on validation performance
        scheduler.step(avg_val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == 0:
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

# Train the monospace font renderer
def train_string_renderer():
    print("Loading monospace font dataset...")

    # Load the pre-generated monospace font dataset
    dataset = create_string_dataset(
        num_samples=5000,  # Use up to 5000 samples for training
        samples_dir="train_monospace_input"
    )

    print("Training attention-based monospace font renderer...")
    # Initialize model with sheet dimensions and appropriate upscaling
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
        # Larger batch size for GPU training
        batch_size = 64
    elif torch.backends.mps.is_available():
        # Moderate batch size for MPS
        batch_size = 32
    else:
        # Smaller batch size for CPU
        batch_size = 16
    
    print(f"Using batch size {batch_size} for sheet dimensions {SHEET_WIDTH}x{SHEET_HEIGHT}")
    
    # Train the model
    model = train_attention_model(
        model,
        dataset,
        num_epochs=100,
        batch_size=batch_size,
        early_stopping_patience=15,  # Patience to see learning curve
        validation_split=0.1  # 10% validation data
    )

    return model

def save_model(model, filename="font_renderer.pth"):
    """Save model weights to a file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename="monospace_renderer.pth"):
    """Load model weights from a file"""
    # Initialize model with global constants from our monospace font setup
    model = AttentionFontRenderer(
        max_length=MAX_CHARS_PER_SHEET,
        sheet_height=SHEET_HEIGHT,
        sheet_width=SHEET_WIDTH,
        scale_factor=DEFAULT_SCALE_FACTOR
    )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(filename, map_location=device))
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Model loaded from {filename}")
    except Exception as e:
        print(f"Error loading model from {filename}: {e}")
        print("This could be due to changes in model architecture or sheet dimensions.")
        raise
        
    return model

if __name__ == '__main__':
    import sys

    # Ensure output directory exists at start
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Test strings for the monospace font model (higher resolution allows more characters per line)
    test_strings = [
        "HELLO LEANN I LOVE YOU SO MUCH I HOPE YOU HAVE A GREAT DAY",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "WWWWWWWWWWWWWWWWWWWW",  # Width test (repeating wide character)
        "IIIIIIIIIIIIIIIIIIII",  # Width test (repeating narrow character)
        "ALTERNATING CASE TEST   SPACES",  # Spacing test 
        "CLAUDE IS RENDERING MONOSPACE FONTS",
        "ZYXWVUTSRQPONMLKJIHGFEDCBA",  # Reverse alphabet
        "AEIOU BCDFGHJKLMNPQRSTVWXYZ",  # Vowels and consonants grouped
        "EXACTLY TWENTY CHARS",  # Boundary test
        "                    ",
    ]
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            # First check if the monospace dataset exists
            if not os.path.exists("train_monospace_input"):
                print("Error: Monospace font dataset not found.")
                print("Please run generate_font_bitmap.py first to create the dataset.")
                sys.exit(1)
                
            # Train mode: train a new model and save it
            model = train_string_renderer()
            save_model(model, filename="monospace_renderer.pth")

            # Render test strings
            render_strings(model, test_strings, output_dir=OUTPUT_DIR)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options: --train")
            sys.exit(1)
    else:
        # Render mode: load model if available, otherwise train first
        model_filename = "monospace_renderer.pth"
        if os.path.exists(model_filename):
            model = load_model(filename=model_filename)
        else:
            # Check if the dataset exists
            if not os.path.exists("train_monospace_input"):
                print("Error: Monospace font dataset not found.")
                print("Please run generate_font_bitmap.py first to create the dataset.")
                sys.exit(1)
                
            print("No saved model found. Training a new model...")
            model = train_string_renderer()
            save_model(model, filename=model_filename)

        # Render strings - edit this list to change what gets rendered
        render_strings(model, test_strings, output_dir=OUTPUT_DIR)
