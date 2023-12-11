import torch
import torch.nn.functional as F
from src.models import DCGenerator, DCDiscriminator


def main():
    # Define parameters
    z_dim = 100  # Dimension of the noise vector
    num_samples = 2  # Number of samples to generate
    num_classes = 10  # Number of classes in CIFAR-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generator and Discriminator
    G = DCGenerator(z_dim).to(device)
    D = DCDiscriminator().to(
        device
    )  # Adjust based on your actual Discriminator's initialization

    G.eval()  # Set the generator to evaluation mode
    D.eval()  # Set the discriminator to evaluation mode

    # Create random noise as input for G
    dummy_noise = torch.randn(num_samples, z_dim).to(device)

    # Generate random labels and convert them to one-hot encoding
    labels_fake = torch.randint(low=0, high=num_classes, size=(num_samples,)).to(device)
    labels_fake_onehot = (
        F.one_hot(labels_fake, num_classes=num_classes).float().to(device)
    )

    # Generate fake images
    gen_output = G(dummy_noise, labels_fake_onehot)[0]

    # Test Discriminator on fake images
    disc_output_fake = D(gen_output, labels_fake_onehot)
    print("Discriminator output for fake images:", disc_output_fake)


if __name__ == "__main__":
    main()
