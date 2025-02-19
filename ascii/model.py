# Goal: Map one ASCII character to its 8x6 bitmap.

# Data Preparation:
# Create a dataset where each input is an ASCII character (integer 0–127), and the output is its 8x6 bitmap (flattened to 48 binary pixels).
# Use your custom font to generate these bitmaps.
# Model Architecture:
# Input: One-hot encoded ASCII character (shape = (128,)).
# Layers:
# Dense(64, activation='relu')
# Dense(48, activation='sigmoid') (output layer).
# Loss: BinaryCrossentropy (each pixel is a binary classification).
# Training:
# Train until loss nears zero (exact reconstruction is possible here).
# Test with characters like 'A', 'B', etc., to validate.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import chars  # using the custom font from ascii/chars.py

# Define the network: one hidden layer with ReLU and an output layer with Sigmoid.
class AsciiModel(nn.Module):
    def __init__(self):
        super(AsciiModel, self).__init__()
        self.fc1 = nn.Linear(128, 48)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x

# Create a dataset based on the custom font.
# For each character in chars, the input is a one-hot vector (length 128)
# and the target is its 8x6 bitmap (flattened into 48 binary values).
def create_dataset():
    inputs = []
    targets = []
    for char, bitmap in chars.chars.items():
        one_hot = [0] * 128
        one_hot[ord(char)] = 1  # set the position corresponding to the ASCII code
        inputs.append(one_hot)
        targets.append(bitmap)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    return data.TensorDataset(inputs_tensor, targets_tensor)

# Simple training loop using the entire dataset as a single batch.
def train_model(model, dataset, num_epochs=10000, lr=0.01):
    dataloader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        if loss.item() < 1e-4:
            print(f"Converged at epoch {epoch}, Loss: {loss.item()}")
            break
    return model

# Helper function to print the 8x6 bitmap.
def print_bitmap(bitmap):
    # bitmap is expected to be a flat list or 1D tensor of length 48.
    for i in range(8):
        row = bitmap[i * 6:(i + 1) * 6]
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().numpy().tolist()
        # Use '#' for "on" pixels (>= 0.5) and ' ' for "off".
        line = ''.join(['#' if p >= 0.5 else ' ' for p in row])
        print(line)

if __name__ == '__main__':
    # Build the dataset and create the model.
    dataset = create_dataset()
    model = AsciiModel()

    print("Training model...")
    model = train_model(model, dataset)

    # Test the trained model with some characters.
    print("\nTesting the model:")
    test_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
    for ch in test_chars:
        one_hot = [0] * 128
        one_hot[ord(ch)] = 1
        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
        pred = model(x).squeeze()  # shape (48,)
        print(f"\nCharacter: {ch}")
        print_bitmap(pred)

