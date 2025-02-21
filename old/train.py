import torch
import torch.nn as nn
import torch.optim as optim
from model import TextToImage
import matplotlib.pyplot as plt
import os
from PIL import Image
from data import texts
from torchvision import transforms

# Hyperparameters (adjust as needed)
vocab_size = 256      # e.g., extended ASCII; adjust for your tokenization scheme
emb_dim = 128         # embedding dimension for text tokens
nhead = 4             # number of attention heads
num_layers = 2        # number of transformer layers
latent_dim = 256      # dimension of the intermediate latent vector
image_width = 400     # width of the generated image
image_height = 800    # height of the generated image
learning_rate = 1e-1  # lower learning rate for more stable training
num_epochs = 20      # train for more epochs
batch_size = 32
seq_length = 100       # maximum number of tokens in your input text

# Initialize the model, loss, and optimizer
model = TextToImage(vocab_size, emb_dim, nhead, num_layers, latent_dim, image_width, image_height)
criterion = nn.L1Loss()  # Using pixel-level loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Optional: use a scheduler to reduce the learning rate every 10 epochs by a factor of 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, texts, images_dir, seq_length, image_width, image_height):
        self.texts = texts
        self.images_dir = images_dir
        self.seq_length = seq_length
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),  # (height, width)
            transforms.ToTensor(),  # converts image to tensor with values in [0, 1]
        ])

    def tokenize_text(self, text):
        # Convert each character to its ASCII code
        tokens = [ord(c) for c in text]
        if len(tokens) < self.seq_length:
            tokens.extend([0] * (self.seq_length - len(tokens)))
        else:
            tokens = tokens[:self.seq_length]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize_text(text)
        # Build filename: images are 1-indexed and padded with a zero if needed.
        filename = os.path.join(self.images_dir, f"image_{idx+1:02d}.png")
        image = Image.open(filename).convert('L')
        image = self.transform(image)
        return tokens, image

# Create dataset and DataLoader
dataset = TextImageDataset(texts, "train_data", seq_length, image_width, image_height)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0
    for tokens, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    scheduler.step()  # step the scheduler after each epoch

# --- Testing after training ---
# Set model to evaluation mode and import matplotlib for visualization.
model.eval()

def tokenize_text(text, seq_length):
    """
    Converts a string into token indices using the ASCII values.
    Pads with 0 if the text is shorter than seq_length,
    or truncates if longer.
    """
    tokens = [ord(c) for c in text]
    if len(tokens) < seq_length:
        tokens.extend([0] * (seq_length - len(tokens)))
    else:
        tokens = tokens[:seq_length]
    return torch.tensor(tokens)

# Define a test input. For example, we'll test with "Hello, world!"
test_text = "Image 9: hello world lol"
test_tokens = tokenize_text(test_text, seq_length).unsqueeze(0)  # shape: (1, seq_length)

with torch.no_grad():
    output_img = model(test_tokens)

# Convert output image tensor to a numpy array for visualization.
output_img_np = output_img.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(output_img_np, cmap='gray')
plt.title(f"Test output for: {test_text}")
plt.show()
