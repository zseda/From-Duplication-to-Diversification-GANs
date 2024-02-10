import torch
from src.models import DiscriminatorCustom as Discriminator

# Initialize the discriminator
D = Discriminator()

# Create a dummy input tensor of size 32x32 (CIFAR-10 size)
dummy_input = torch.rand(size=(2, 3, 32, 32))  # Batch size of 1 for simplicity

# Forward the dummy input through the base model
with torch.no_grad():  # No need to compute gradients for this operation
    base_model_output = D(dummy_input)
    print("Base model output size before flattening:", base_model_output.shape)
