import torch
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
from loguru import logger
from src.data import get_cifar10_dataloader
import wandb
from datetime import datetime
from src.models import (
    VanillaGenerator,
    VanillaDiscriminator,
)  # Ensure these are correctly defined


# Function to set random seeds
def set_random_seeds(seed_value=42):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)


# Set random seeds for reproducibility
set_random_seeds(seed_value=42)


class GAN(LightningModule):
    def __init__(self, generator, discriminator, latent_dim=100):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.validation_z = torch.randn(8, latent_dim)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch  # This should unpack the batch into images and labels
        # Check if real_imgs is a list and convert it to a tensor
        real_imgs = real_imgs.to(self.device)

        batch_size = real_imgs.size(0)

        # Generator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()

        g_loss = self.adversarial_loss(
            self.discriminator(generated_imgs),
            torch.ones(batch_size, 1, device=self.device),
        )
        self.manual_backward(g_loss)
        self.opt_g.step()

        # Discriminator update
        self.opt_d.zero_grad()
        self.opt_g.zero_grad()
        real_loss = self.adversarial_loss(
            self.discriminator(real_imgs), torch.ones(batch_size, 1, device=self.device)
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(generated_imgs.detach()),
            torch.zeros(batch_size, 1, device=self.device),
        )
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        self.opt_d().step()

        if batch_idx % 50 == 0:
            # Log losses
            self.logger.experiment.log(
                {
                    "losses/d_fake": fake_loss,
                    "losses/d_real": real_loss,
                    "losses/g": g_loss,
                }
            )

        if batch_idx % 250 == 0:
            with torch.no_grad():
                # Generate images for logging
                generated_imgs = self(self.validation_z.to(self.device))
                img_grid_fake = make_grid(generated_imgs, normalize=True)
                img_grid_real = make_grid(batch[0], normalize=True)

                # Log generated and real images
                self.logger.experiment.log(
                    {
                        "images/generated": [
                            wandb.Image(img_grid_fake, caption="Generated Images")
                        ],
                        "images/real": [
                            wandb.Image(img_grid_real, caption="Real Images")
                        ],
                    }
                )

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        # Get both optimizers
        self.opt_g = opt_g
        self.opt_d = opt_d
        return opt_d, opt_g


# Assuming get_cifar10_dataloader is defined and returns a DataLoader
dataloader = get_cifar10_dataloader(batch_size=128, num_workers=8)[0]


wandb_logger = WandbLogger(project="Vanilla-GAN", log_model="all")

# Initialize the GAN module with your generator and discriminator
model = GAN(VanillaGenerator(), VanillaDiscriminator())

# Initialize ModelCheckpoint callback to save the last model
checkpoint_callback = ModelCheckpoint(
    dirpath="./model_checkpoints/",
    filename="Vanilla-GAN-{epoch:02d}",
    save_top_k=-1,  # Save all models
    every_n_epochs=25,  # Save every epoch
    verbose=True,
)

# Check for GPU availability
gpus = 1 if torch.cuda.is_available() else 0

# Trainer
trainer = Trainer(
    max_epochs=200,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    devices=gpus,
)

trainer.fit(model, dataloader)
