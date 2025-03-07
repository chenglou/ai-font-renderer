"""
This file implements an attention-based neural network for rendering ASCII text as bitmap font images.

The model uses self-attention mechanisms that allow characters to influence each other's rendering,
creating a more coherent font appearance across a string. It includes a complete pipeline for:
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
  - Focal loss works better than standard BCE loss for this task (confirmed)
  - Early stopping based on validation helps prevent overfitting (confirmed)
  - A balanced model size with moderate embedding dimensions (80) works well
  - Both validation and regularization are important for generalization
  - Simpler architectures should be preferred when they perform comparably

Performance optimizations:
  - Hardware acceleration with MPS (Metal Performance Shaders) gives ~60% speedup on M-series Macs
  - Larger batch sizes (128) improve training efficiency and output quality
  - Default learning rate (0.001) with batch size 128 provides best speed/quality balance
  - Training time ~4-5 minutes on M2 Pro with these optimizations (vs 26+ minutes without)
  - Reduced model complexity (removing FC layers) further improves training efficiency

These observations are based on experimentation with this specific task and dataset.
Different font styles or character sets might require different approaches.

The model supports sheet-based rendering with a fixed output size (40x120 pixels by default).
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

# Use MPS (Metal Performance Shaders) for M-series Mac if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Modified model with single attention layer to test its importance
class AttentionFontRenderer(nn.Module):
    def __init__(self, max_length=100, sheet_height=40, sheet_width=120):
        super().__init__()
        self.max_length = max_length
        self.sheet_height = sheet_height
        self.sheet_width = sheet_width
        self.sheet_size = sheet_height * sheet_width

        # Keep the same embedding size
        self.embedding = nn.Embedding(128, 80)
        self.embedding_dropout = nn.Dropout(0.1)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, 80))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Single attention layer for testing
        self.attention = nn.MultiheadAttention(embed_dim=80, num_heads=4, dropout=0.1)
        self.layer_norm = nn.LayerNorm(80)

        # Simplified processing network - removed fc2 layer
        self.fc1 = nn.Linear(80, 160)
        self.dropout1 = nn.Dropout(0.15)
        # fc2 layer removed
        self.fc3 = nn.Linear(160 * max_length, self.sheet_size)

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

        # Generate the entire sheet bitmap
        sheet = self.output_activation(self.fc3(x))  # [batch_size, sheet_size]

        # Reshape to proper dimensions
        sheet = sheet.view(batch_size, self.sheet_height, self.sheet_width)  # [batch_size, 40, 120]

        return sheet

# Generate random strings from the available characters
def generate_random_string(length):
    available_chars = list(chars.chars.keys())
    return ''.join(random.choice(available_chars) for _ in range(length))

# Create a dataset of text sheets and save sample images to a folder
def create_string_dataset(num_samples=1000, min_length=20, max_length=100,
                         sheet_height=40, sheet_width=120, char_height=8, char_width=6,
                         save_samples=False, samples_dir="train_input", num_samples_to_save=10):
    # Reset random seed for reproducible dataset generation
    random.seed(SEED)

    # Calculate how many characters per row and maximum rows
    chars_per_row = sheet_width // char_width
    max_rows = sheet_height // char_height

    # Maximum characters per sheet
    max_chars_per_sheet = chars_per_row * max_rows

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

        # Position characters in the sheet (monospace layout)
        char_idx = 0
        for row in range(max_rows):
            if char_idx >= len(string):
                break

            for col in range(chars_per_row):
                if char_idx >= len(string):
                    break

                # Get character bitmap
                char_bitmap = chars.chars[string[char_idx]]

                # Calculate position in the sheet
                y_start = row * char_height
                x_start = col * char_width

                # Place character bitmap in the sheet
                for y in range(char_height):
                    for x in range(char_width):
                        bitmap_idx = y * char_width + x
                        if bitmap_idx < len(char_bitmap):  # Safety check
                            all_targets[sample_idx, y_start + y, x_start + x] = char_bitmap[bitmap_idx]

                char_idx += 1

        # Save this sample as an image if requested
        if save_samples and sample_idx < num_samples_to_save:
            # Convert binary sheet to image
            img = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
            for y in range(sheet_height):
                for x in range(sheet_width):
                    if all_targets[sample_idx, y, x] >= 0.5:
                        img[y, x] = 0

            # Scale up the image for better visibility
            scale = 4
            img_scaled = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

            # Convert to PIL Image and save
            pil_img = Image.fromarray(img_scaled)
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
    # Split dataset into training and validation
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
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
        worker_init_fn=lambda id: random.seed(SEED + id)
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g
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

            # Move inputs and targets to device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            outputs = model(batch_inputs)
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
                # Move inputs and targets to device
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                # Forward pass
                outputs = model(batch_inputs)
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
def render_strings(model, strings, output_dir="train_test_simplified_fc"):
    """Render a list of strings as BMP images"""
    # Fixed parameters
    scale = 4

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

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
            sheet = model(x).squeeze(0)  # shape [sheet_height, sheet_width]

        # Convert prediction to numpy array
        if isinstance(sheet, torch.Tensor):
            sheet = sheet.detach().cpu().numpy()

        # Create a new image (with white background)
        img = np.ones((model.sheet_height, model.sheet_width), dtype=np.uint8) * 255

        # Fill in the image with the predicted bitmap
        for row in range(model.sheet_height):
            for col in range(model.sheet_width):
                # Set pixel black (0) if the value is >= 0.5; white otherwise
                if sheet[row, col] >= 0.5:
                    img[row, col] = 0

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

# Train the sheet-based renderer
def train_string_renderer(generate_only=False):
    print("Creating sheet dataset...")
    # Create dataset with appropriate parameters for our sheet size
    sheet_height = 40
    sheet_width = 120
    char_height = 8
    char_width = 6

    # Calculate how many characters can fit
    chars_per_row = sheet_width // char_width
    max_rows = sheet_height // char_height
    max_chars = chars_per_row * max_rows

    # Create the dataset and save samples to the train_input folder
    dataset = create_string_dataset(
        num_samples=5000,  # Increased to 5000 samples for better learning
        min_length=10,
        max_length=max_chars,
        sheet_height=sheet_height,
        sheet_width=sheet_width,
        char_height=char_height,
        char_width=char_width,
        save_samples=True,  # Save sample images
        samples_dir="train_input",
        num_samples_to_save=10  # Save 10 samples for reference
    )

    # If generate_only flag is set, return without training
    if generate_only:
        print("Dataset generation complete. Skipping training.")
        return None

    print("Training attention-based sheet renderer...")
    # Initialize model with sheet dimensions
    model = AttentionFontRenderer(
        max_length=max_chars,
        sheet_height=sheet_height,
        sheet_width=sheet_width
    )
    model = model.to(device)  # Move model to MPS device

    # Train with same settings but using simplified model
    output_dir = "train_test_simplified_fc"
    os.makedirs(output_dir, exist_ok=True)
    model = train_attention_model(
        model,
        dataset,
        num_epochs=100,
        batch_size=128,  # Larger batch size for better speed/quality balance
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
    # Use the same dimensions as in training
    sheet_height = 40
    sheet_width = 120

    # Calculate maximum characters
    char_height = 8
    char_width = 6
    chars_per_row = sheet_width // char_width
    max_rows = sheet_height // char_height
    max_chars = chars_per_row * max_rows

    # Initialize model with correct dimensions
    model = AttentionFontRenderer(
        max_length=max_chars,
        sheet_height=sheet_height,
        sheet_width=sheet_width
    )
    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filename}")
    return model

if __name__ == '__main__':
    import sys

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
        "EXACTLY TWENTY CHARS"  # Boundary test
    ]
    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            # Train mode: train a new model and save it
            model = train_string_renderer()
            save_model(model)

            # Render test strings
            render_strings(model, test_strings)
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
        render_strings(model, test_strings)
