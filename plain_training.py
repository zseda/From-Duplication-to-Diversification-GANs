import torch
import torchvision
import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from pathlib import Path
from src.models import Generator, DiscriminatorCustom as Discriminator
from src.data import get_single_cifar10_dataloader as get_cifar10_dataloader
from typer import Typer, Option

app = Typer()


class GAN(pl.LightningModule):
    def __init__(self):

        super(GAN, self).__init__()

        # create generator
        self.generator = Generator(self.device).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.target_class = 4
        # Initialize generator with dummy inputs to set up layers
        dummy_noise = torch.rand(size=(1, 56, 2, 2), device=self.device)
        dummy_images = torch.rand(size=(1, 3, 32, 32), device=self.device)
        self.generator(dummy_noise, dummy_images)

    def forward(self, noise, images):
        return self.generator(noise, images)

    def adversarial_loss(self, y_hat, y):
        return torch.nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        noise = torch.rand(real_imgs.size(0), 56, 2, 2, device=self.device)
        fake_imgs = self(noise, real_imgs)

        # Train discriminator
        if optimizer_idx == 0:
            real_loss = self.adversarial_loss(
                self.discriminator(real_imgs),
                torch.ones(real_imgs.size(0), 1, device=self.device),
            )
            fake_loss = self.adversarial_loss(
                self.discriminator(fake_imgs.detach()),
                torch.zeros(real_imgs.size(0), 1, device=self.device),
            )
            d_loss = (real_loss + fake_loss) / 2
            self.log(
                "d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            # Log losses every 50 batches
            if batch_idx % 50 == 0:
                self.logger.experiment.log(
                    {
                        "losses/d_real": real_loss,
                        "losses/d_fake": fake_loss,
                        "losses/d": d_loss,
                    }
                )
            return d_loss

        # Train generator
        if optimizer_idx == 1:
            g_loss = self.adversarial_loss(
                self.discriminator(fake_imgs),
                torch.ones(real_imgs.size(0), 1, device=self.device),
            )
            self.log(
                "g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if batch_idx % 250 == 0:
            with torch.no_grad():
                # Log generated images
                img_grid_fake = torchvision.utils.make_grid(
                    fake_imgs[:16], normalize=True
                )
                self.logger.experiment.log(
                    {
                        "images/generated": [
                            wandb.Image(img_grid_fake, caption="Generated Images")
                        ]
                    }
                )

                # Log real images
                img_grid_real = torchvision.utils.make_grid(
                    real_imgs[:16], normalize=True
                )
                self.logger.experiment.log(
                    {"images/real": [wandb.Image(img_grid_real, caption="Real Images")]}
                )

        return g_loss

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def train_dataloader(self):
        return get_cifar10_dataloader(self.target_class)

    def save_model_as_onnx(self, epoch, run_dir):
        run_dir = Path(run_dir)
        dummy_noise = torch.randn(1, 56, 2, 2, device=self.device)
        dummy_images = torch.randn(1, 3, 32, 32, device=self.device)
        onnx_file_path = run_dir / f"generator_epoch_{epoch}.onnx"
        torch.onnx.export(self.generator, (dummy_noise, dummy_images), onnx_file_path)
        logger.info(
            f"Model at epoch {epoch} has been converted to ONNX and saved to {onnx_file_path}"
        )

    def on_epoch_end(self):
        current_epoch = self.current_epoch
        if (current_epoch + 1) % 50 == 0:
            self.save_model_as_onnx(current_epoch + 1, self.logger.experiment.dir)


@app.command()
def train(
    max_epochs: int = Option(500),
    wandb_run_name: str = Option("GAN-Simple-BCE-Experiment"),
):
    current_time = datetime.now()
    session_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    full_wandb_run_name = f"{wandb_run_name}-{session_name}"
    # Create a directory for this run using pathlib
    run_dir = Path(f"./runs/{full_wandb_run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Check for GPU availability
    gpus = 1 if torch.cuda.is_available() else 0

    # Weights & Biases setup for online-only logging
    wandb.init(
        project="GAN-CIFAR10",
        name=full_wandb_run_name,
        settings=wandb.Settings(mode="online"),
    )

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        devices=gpus,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=run_dir,
                every_n_epochs=50,
                filename="{epoch:02d}-{g_loss:.2f}-{d_loss:.2f}",
            )
        ],
    )
    # Initialize your GAN model
    gan = GAN()

    # Start the training process
    trainer.fit(gan)

    # Finish the wandb run
    wandb.finish()

    logger.info("Finished training!")


if __name__ == "__main__":
    app()
