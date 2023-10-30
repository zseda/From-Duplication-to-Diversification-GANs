import torch
from src.models import Generator
from src.models import Discriminator


def main():
    # Test Generator
    G = Generator()
    dummy_img = torch.rand(size=(2, 3, 224, 224))
    dummy_noise = torch.rand(size=(2, 112, 14, 14))
    gen_output = G(dummy_img, dummy_noise)

    print("Generator output shape:", gen_output.shape)

    # Test Discriminator
    D = Discriminator()
    dummy_output = torch.rand(size=(2, 3, 224, 224))
    disc_output = D(dummy_output)

    print("Discriminator output shape:", disc_output.shape)


if __name__ == "__main__":
    main()
