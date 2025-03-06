from torchviz import make_dot
import torch
from model import AttentionFontRenderer  # Updated import to match current architecture

# Create a dummy input that matches the AttentionFontRenderer input shape
# Input is a batch of ASCII character codes: [batch_size, seq_len]
batch_size = 1
seq_len = 10  # Example sequence length
dummy_input = torch.randint(0, 128, (batch_size, seq_len), dtype=torch.long)

# Initialize the model with the same max_length as in training
model = AttentionFontRenderer(max_length=25)

# Do a forward pass
output = model(dummy_input)

# Generate the graph, including the model parameters
graph = make_dot(output, params=dict(model.named_parameters()))

# Render the graph to PNG and clean up the intermediate file
graph.render("AttentionFontRenderer", format="png", cleanup=True)
print("Model visualization saved to AttentionFontRenderer.png")

# Export the model to an ONNX file
torch.onnx.export(
    model,
    dummy_input,
    "attention_font_renderer.onnx",
    input_names=['input'],
    output_names=['output']
)
print("Model exported to attention_font_renderer.onnx")
