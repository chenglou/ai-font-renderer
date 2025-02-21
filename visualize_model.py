from torchviz import make_dot
import torch
from model import AsciiModel  # make sure the import matches your project structure

# Create a dummy input that matches your model's input shape.
dummy_input = torch.randn(1, 128)
model = AsciiModel()

# Do a forward pass.
output = model(dummy_input)

# Generate the graph, including the model parameters.
graph = make_dot(output, params=dict(model.named_parameters()))

# Render the graph to PNG and clean up the intermediate file.
graph.render("AsciiModel", format="png", cleanup=True)

# Export the model to an ONNX file.
torch.onnx.export(
    model,
    dummy_input,
    "ascii_model.onnx",
    input_names=['input'],
    output_names=['output']
)
print("Model exported to ascii_model.onnx")
