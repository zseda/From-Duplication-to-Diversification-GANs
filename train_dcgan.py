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
from src.models import DCGenerator, DCDiscriminator
import timm
import uuid
from loguru import logger


class GAN(pl.LightningModule):
    def __init__(self, z_dim=100, lr=1e-4, start_c_after=15):
        super().__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.lr = lr
        self.start_c_after = start_c_after
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.z_dim = z_dim
        self.lr = lr
        self.start_c_after = start_c_after

        # Model Definitions
        self.C = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=10, in_chans=1
        ).to(self.device)

        self.G = DCGenerator(g_input_dim=z_dim).to(self.device)

        # config D - weight norm
        D_weight_norm = nn.utils.spectral_norm

        # config D - norm
        D_norm = nn.Identity

        # config D - activation
        D_activation = nn.LeakyReLU

        self.D = DCDiscriminator(
            norm=D_norm, weight_norm=D_weight_norm, activation=D_activation
        ).to(self.device)

        self.automatic_optimization = False

        # Loss Functions
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

    def on_epoch_start(self):
        if self.sample_val_images is None:
            self.sample_val_images = next(iter(self.train_dataloader()))[0].to(
                self.device
            )

    def training_step(self, batch, batch_idx):
        """
        -----------
        preparation
        -----------
        """
        # decompose batch data
        img, labels_real = batch
        img = img.to(self.device)
        labels_real = labels_real.to(self.device)
        labels_real_onehot = (
            F.one_hot(labels_real, num_classes=10).float().to(self.device)
        )

        actual_batch_size = img.shape[0]
        # create labels
        y_real = Variable(torch.ones(actual_batch_size, 1).to(self.device))
        y_fake = Variable(torch.zeros(actual_batch_size, 1).to(self.device))

        # generate random noise for G
        z = Variable(torch.randn(actual_batch_size, self.z_dim).to(self.device))

        # create class labels for generator - uniform distribution
        labels_fake = torch.randint(low=0, high=9, size=(actual_batch_size,))

        # one-hot encode class labels
        labels_fake_onehot = (
            F.one_hot(labels_fake, num_classes=10).float().to(self.device)
        )
        # generate images for D
        x_fake, x_fake_logits = self.G(z, labels_fake_onehot)
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        self.opt_c.zero_grad()
        """
            -------
            train C
            -------
        """
        C_out = self.C(img)
        C_loss = self.criterion_classification(C_out, labels_real_onehot)
        C_loss.backward()
        self.opt_c.step()

        """
            -------
            train D 
            -------
        """
        # Discriminator update

        D_out_real = self.D(img, labels_real_onehot)
        D_real_loss = self.criterion(D_out_real, y_real)

        D_out_fake = self.D(x_fake, labels_fake_onehot)
        D_fake_loss = self.criterion(D_out_fake, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = (D_real_loss + D_fake_loss) / 2.0
        D_loss.backward()
        self.opt_d.step()

        """
            ------- 
            train G
            -------
        """
        # reset gradients
        self.opt_d.zero_grad()
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()

        # generate images via G
        # create labels for testing generator
        # convert to one hot encoding
        z = Variable(torch.randn(actual_batch_size, self.z_dim).to(self.device))

        G_output, G_output_logits = self.G(z, labels_fake_onehot)
        D_out = self.D(G_output, labels_fake_onehot)
        G_disc_loss = self.criterion(D_out, y_real)

        # test generated images with classifier
        if self.current_epoch > self.start_c_after:
            C_out = self.C(G_output)
            G_classification_loss = self.criterion_classification(
                C_out, labels_fake_onehot
            )
            G_loss = G_disc_loss + 0.3 * G_classification_loss
        else:
            G_loss = G_disc_loss

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.opt_g.step()

        # TODOLog losses
        # TODOLog images

    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_c = torch.optim.Adam(self.C.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.opt_c = opt_c
        return opt_g, opt_d, opt_c

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_dataloader(batch_size=128, num_workers=8)[0]


def main(
    root_path: str = typer.Option("."),
    epochs: int = typer.Option(100),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    z_dim: int = typer.Option(100),
    start_c_after: int = typer.Option(15),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_size: int = typer.Option(48000),
):
    # Initialize WandB
    wandb_logger = WandbLogger(name="GAN_Experiment", project="GAN_Project")

    # Initialize our model
    model = GAN(z_dim=z_dim, lr=lr, start_c_after=start_c_after)

    # Trainer
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        max_epochs=500, accelerator="cpu", devices=1, logger=wandb_logger
    )
    trainer.fit(model)


if __name__ == "__main__":
    typer.run(main)
