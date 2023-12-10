import typer
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.data import get_cifar10_dataloader as get_dataloader
from src.models import DCGenerator, DCDiscriminator, weights_init_normal
import timm
import uuid
from loguru import logger


class DCGAN(pl.LightningModule):
    def __init__(self, g_input_dim=100, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.generator = DCGenerator(g_input_dim=g_input_dim)
        self.discriminator = DCDiscriminator(
            norm=nn.Identity,
            weight_norm=nn.utils.spectral_norm,
            activation=nn.LeakyReLU,
        )
        self.criterion = nn.BCELoss()
        self.g_input_dim = g_input_dim
        self.lr = lr

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, labels = batch

        # Train Generator
        if optimizer_idx == 0:
            z = torch.randn(
                real_imgs.shape[0], self.g_input_dim, 1, 1, device=self.device
            )
            fake_imgs, _ = self(z, labels)
            fake_validity = self.discriminator(fake_imgs, labels)
            g_loss = self.criterion(fake_validity, torch.ones_like(fake_validity))
            self.log("loss/generator", g_loss)
            return g_loss

        # Train Discriminator
        if optimizer_idx == 1:
            fake_imgs, _ = self(
                torch.randn(
                    real_imgs.shape[0], self.g_input_dim, 1, 1, device=self.device
                ),
                labels,
            ).detach()
            real_validity = self.discriminator(real_imgs, labels)
            fake_validity = self.discriminator(fake_imgs, labels)
            real_loss = self.criterion(real_validity, torch.ones_like(real_validity))
            fake_loss = self.criterion(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2
            self.log("loss/discriminator", d_loss)

            # Log images and losses using WandB
            if batch_idx % 50 == 0:
                img_grid = torchvision.utils.make_grid(fake_imgs, normalize=True)
                self.logger.experiment.log(
                    {
                        "images/generated": [
                            wandb.Image(img_grid, caption="Generated Images")
                        ],
                    }
                )
            # Log real images
            img_grid_real = torchvision.utils.make_grid(real_imgs, normalize=True)
            self.logger.experiment.log(
                {"images/real": [wandb.Image(img_grid_real, caption="Real Images")]}
            )

            return {
                "loss": g_loss,
                "log": {"loss_generator": g_loss},
            }

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        return [opt_g, opt_d], []

    def train_dataloader(self):
        # Load your data here using the get_dataloader function
        return get_dataloader(
            batch_size=64, num_workers=4, dataset_size=48000
        )  # Replace with actual parameters


def main(epochs: int = 100, g_input_dim: int = 100, lr: float = 1e-4):
    wandb_logger = WandbLogger(project="dcgan")
    model = DCGAN(g_input_dim=g_input_dim, lr=lr)
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    trainer.fit(model)


if __name__ == "__main__":
    typer.run(main)
