import torch
import torch.nn as nn
import torch.nn.functional as F  # <-- import functional for one_hot

class TextToImage(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, latent_dim, image_width, image_height):
        """
        vocab_size  : number of tokens (e.g., 256 for ASCII)
        emb_dim     : desired embedding dimension (if different from vocab_size, a projection is applied)
        nhead       : number of heads in transformer encoder
        num_layers  : number of transformer encoder layers
        latent_dim  : size of the intermediate latent vector (after aggregation)
        image_width : desired width of generated image
        image_height: desired height of generated image
        """
        super(TextToImage, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.vocab_size = vocab_size   # store vocab_size for one-hot conversion
        self.emb_dim = emb_dim          # store emb_dim for forward pass usage

        # Instead of a learned nn.Embedding, we use a fixed one-hot encoding.
        # Optionally, if emb_dim is not equal to vocab_size, add a fixed projection.
        if emb_dim != vocab_size:
            self.proj = nn.Linear(vocab_size, emb_dim)
        else:
            self.proj = None

        # Transformer encoder expects an input dimension of emb_dim.
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP that processes the aggregated transformer output to generate image pixels.
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, image_width * image_height)  # flatten image pixels
        )

    def forward(self, x):
        # x: (batch_size, seq_length) with ASCII values
        # Convert directly to float and normalize (values between 0 and 1)
        x = x.float() / 255.0  # normalization might help training
        # Make it (batch_size, seq_length, 1)
        x = x.unsqueeze(-1)
        # To match the transformer d_model, duplicate the value across emb_dim dimensions
        x = x.repeat(1, 1, self.emb_dim)
        x = x.transpose(0, 1)  # shape: (seq_length, batch_size, emb_dim)
        # Then proceed with transformer and the rest of the model as before.
        encoded = self.transformer_encoder(x)
        aggregated = encoded.mean(dim=0)
        img_flat = self.fc(aggregated)
        img = img_flat.view(-1, 1, self.image_height, self.image_width)
        return img
