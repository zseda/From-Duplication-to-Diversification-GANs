import torch
from src.models import Generator
from src.models import Discriminator


def main():
    # Test Generator
    G = Generator("cpu")
    dummy_img = torch.rand(size=(2, 3, 32, 32))

    dummy_noise = torch.rand(size=(2, 56, 2, 2))
    gen_output = G(dummy_img, dummy_noise)

    print("Generator output shape:", gen_output.shape)

    # Test Discriminator
    D = Discriminator()
    dummy_output = torch.rand(size=(2, 3, 32, 32))
    disc_output = D(dummy_output)

    print("Discriminator output shape:", disc_output.shape)


if __name__ == "__main__":
    main()
