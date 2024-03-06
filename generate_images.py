import torch
import os
from torchvision.utils import save_image
from src.models import Generator2 as Generator


def load_generator(checkpoint_path, device):
    generator = Generator(device=device)  # Initialize the generator
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    return generator


def generate_images(generator, num_images, device, batch_size=10):
    save_dir = "CIFAR10_class4_customgan_generated"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(0, num_images, batch_size):
        # Prepare dummy input images and noise tensors
        dummy_images = torch.rand(batch_size, 3, 32, 32, device=device)
        noise = torch.rand(batch_size, 56, 2, 2, device=device)

        with torch.no_grad():
            generated_images = generator(dummy_images, noise)

        for j, image in enumerate(generated_images):
            save_path = os.path.join(save_dir, f"image_{i+j+1}.png")
            save_image(image, save_path)

    print(f"Generated {num_images} images in '{save_dir}'")


if __name__ == "__main__":
    device = torch.device("cpu")  # Use 'cuda' if GPU is available
    checkpoint_path = "/home/zeynep/repos/github/From-Duplication-to-Diversification-GANs/lx5a7e0y/generator_175.pt"  # Update with actual path
    generator = load_generator(checkpoint_path, device)
    generate_images(generator, 5000, device)  # Generate 5000 images
