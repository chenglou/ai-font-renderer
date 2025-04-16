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
  - LEARNED positional encodings are CRUCIAL for this task - fixed sinusoidal encodings and RoPE (Rotary Position Embeddings) both failed completely (99% white output)
  - This is likely because font rendering requires understanding 2D layout which only learned positional encodings can capture for this task
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
from helpers import render_strings, save_model, load_model, load_string_dataset, MODEL_FILENAME

SHEET_HEIGHT = 80
SHEET_WIDTH = 240
MAX_CHARS_PER_SHEET = 100
NUM_SAMPLES = 150000

# Output directory for rendered test strings with timestamp
import datetime
OUTPUT_DIR = "train_output_" + datetime.datetime.now().strftime("%m_%d_%H_%M_%S")

# Training hyperparameters
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 70
VALIDATION_SPLIT = 0.2
WEIGHT_DECAY = 0.0005
EMBEDDING_DIM = 32
DROPOUT_RATE = 0.2
NUM_ATTENTION_HEADS = 4
SCHEDULER_PATIENCE = 20
SCHEDULER_FACTOR = 0.7
MIN_LEARNING_RATE = 1e-6

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# Test strings for model evaluation
test_strings = [
    "HELLO LEANN I LOVE YOU SO MUCH I HOPE YOU HAVE A GREAT DAY",
    "TWO WORLDS ONE FAMILY TRUST YOUR HEART LET FATE DECIDE TO GUIDE THESE LIVES WE SEE",
    "A PARADISE UNTOUCHED BY MAN WITHIN THIS WORLD BLESSED WITH LOVE A SIMPLE LIFE THEY LIVE IN PEACE",
    "SOFTLY TREAD THE SAND BELOW YOUR FEET NOW TWO WORLDS ONE FAMILY TRUST YOUR HEART LET FATE",
    "BENEATH THE SHELTER OF THE TREES ONLY LOVE CAN ENTER HERE A SIMPLE LIFE THEY LIVE IN PEACE",
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

class AttentionFontRenderer(nn.Module):
    def __init__(self, max_length=MAX_CHARS_PER_SHEET):
        super().__init__()
        self.max_length = max_length

        # Use global embedding dimension
        self.embedding_dim = EMBEDDING_DIM
        self.embedding = nn.Embedding(128, self.embedding_dim)
        self.embedding_dropout = nn.Dropout(DROPOUT_RATE)

        # Learned positional encoding with reduced dimension
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, self.embedding_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Single attention layer with reduced dimension and increased dropout
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=NUM_ATTENTION_HEADS, dropout=DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        # Processing network (simplified) - adjusted dimensions
        self.fc1 = nn.Linear(self.embedding_dim, 64)  # Reduced from 160
        self.dropout1 = nn.Dropout(DROPOUT_RATE + 0.05)  # Slightly higher dropout for FC layers

        # FC layer outputs feature map directly at full resolution (removed upsampling)
        self.fc_output = nn.Linear(64 * max_length, SHEET_HEIGHT * SHEET_WIDTH)

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

        # Generate full resolution feature map directly
        x = self.fc_output(x)  # [batch_size, SHEET_HEIGHT * SHEET_WIDTH]

        # Reshape to [batch_size, SHEET_HEIGHT, SHEET_WIDTH]
        sheet = x.view(batch_size, SHEET_HEIGHT, SHEET_WIDTH)

        # Apply output activation
        sheet = self.output_activation(sheet)  # [batch_size, SHEET_HEIGHT, SHEET_WIDTH]

        return sheet

# Use the generate_font module to create datasets

# Balanced training function with focal loss and moderate regularization
def train_attention_model(model, dataset, batch_size):
    # Create config file with all parameters
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/config.txt", "w") as config_file:
        config_file.write(f"# Training configuration\n")
        config_file.write(f"num_epochs = {NUM_EPOCHS}\n")
        config_file.write(f"learning_rate = {LEARNING_RATE}\n")
        config_file.write(f"batch_size = {batch_size}\n")
        config_file.write(f"early_stopping_patience = {EARLY_STOPPING_PATIENCE}\n")
        config_file.write(f"validation_split = {VALIDATION_SPLIT}\n")
        config_file.write(f"weight_decay = {WEIGHT_DECAY}\n")
        config_file.write(f"embedding_dim = {EMBEDDING_DIM}\n")
        config_file.write(f"dropout_rate = {DROPOUT_RATE}\n")
        config_file.write(f"num_attention_heads = {NUM_ATTENTION_HEADS}\n")
        config_file.write(f"max_length = {model.max_length}\n")
        config_file.write(f"max_chars_per_sheet = {MAX_CHARS_PER_SHEET}\n")
        config_file.write(f"num_samples = {NUM_SAMPLES}\n")
        config_file.write(f"data_size = {len(dataset)}\n")
        config_file.write(f"random_seed = {SEED}\n")
        config_file.write(f"sheet_height = {SHEET_HEIGHT}\n")
        config_file.write(f"sheet_width = {SHEET_WIDTH}\n")

    # Split the original dataset into training and validation
    orig_dataset_size = len(dataset)
    val_size = int(VALIDATION_SPLIT * orig_dataset_size)
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

    # Optimize DataLoader for GPU usage
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=32,
        pin_memory=True,
        worker_init_fn=lambda id: random.seed(SEED + id)
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
        num_workers=32,
        pin_memory=True
    )

    # Plain MSE loss for grayscale values
    def compute_loss(pred, target):
        return nn.functional.mse_loss(pred, target)

    # AdamW optimizer with minimal weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))

    # Learning rate scheduler with higher patience and smoother reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE
    )

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(NUM_EPOCHS):
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

            loss = compute_loss(outputs, batch_targets)

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

                val_loss = compute_loss(outputs, batch_targets)
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
        if epoch % 5 == 0:  # Print more frequently to track progress better
            status = f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            if is_best:
                status += f" (New Best)"
            print(status)

            # Create a directory for this epoch's outputs
            epoch_dir = f"{OUTPUT_DIR}/epoch_{epoch}"
            # Render test strings to the epoch directory
            render_strings(model, test_strings, output_dir=epoch_dir, sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH, device=device)
        elif is_best:
            print(f"Epoch {epoch}, New best validation loss: {avg_val_loss:.6f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, Best Val Loss: {best_val_loss:.6f}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

    # Ensure best model is loaded
    if best_model_state is not None and patience_counter < EARLY_STOPPING_PATIENCE:
        model.load_state_dict(best_model_state)
        print(f"Training completed, Best Val Loss: {best_val_loss:.6f}")

    # Write training results to file
    final_epoch = epoch + 1 if patience_counter < EARLY_STOPPING_PATIENCE else epoch - patience_counter
    with open(f"{OUTPUT_DIR}/training_results.txt", "w") as results_file:
        results_file.write(f"# Training Results\n")
        results_file.write(f"final_epoch = {final_epoch}\n")
        results_file.write(f"best_validation_loss = {best_val_loss:.6f}\n")
        results_file.write(f"final_learning_rate = {optimizer.param_groups[0]['lr']:.6f}\n")
        results_file.write(f"early_stopped = {patience_counter >= EARLY_STOPPING_PATIENCE}\n")
        results_file.write(f"training_duration_epochs = {final_epoch}\n")
        results_file.write(f"training_completed = {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return model

# Use the imported render_strings function from helpers.py

# Train the sheet-based renderer
def train_string_renderer():
    print("Creating sheet dataset...")

    # Load the dataset from train_input folder
    dataset = load_string_dataset(
        data_dir="train_input",
        num_samples=NUM_SAMPLES,
        sheet_height=SHEET_HEIGHT,
        sheet_width=SHEET_WIDTH
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

# Use the imported save_model and load_model functions from helpers.py

if __name__ == '__main__':
    import sys

    # Ensure output directory exists at start
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            # Train mode: train a new model and save it
            model = train_string_renderer()
            save_model(model)

            # Render test strings
            render_strings(model, test_strings, output_dir=OUTPUT_DIR, sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH, device=device)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options: --train")
            sys.exit(1)
    else:
        # Render mode: load model if available, otherwise train first
        if os.path.exists(MODEL_FILENAME):
            model = load_model(AttentionFontRenderer, MAX_CHARS_PER_SHEET, device=device)
        else:
            print("No saved model found. Training a new model...")
            model = train_string_renderer()
            save_model(model)

        # Render strings - edit this list to change what gets rendered
        render_strings(model, test_strings, output_dir=OUTPUT_DIR, sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH, device=device)
