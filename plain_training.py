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
    def __init__(self, batch_size):

        super(GAN, self).__init__()

        # create generator
        self.generator = Generator(self.device).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.target_class = 4
        self.batch_size = batch_size
        # Initialize generator with dummy inputs to set up layers
        dummy_noise = torch.rand(size=(1, 56, 2, 2), device=self.device)
        dummy_images = torch.rand(size=(1, 3, 32, 32), device=self.device)
        self.generator(dummy_images, dummy_noise)
        self.automatic_optimization = False

    def forward(self, images, noise):
        return self.generator(images, noise)

    def adversarial_loss(self, y_hat, y):
        return torch.nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        noise = torch.rand(real_imgs.size(0), 56, 2, 2, device=self.device)
        fake_imgs = self.generator(real_imgs, noise)

        # Get the optimizers
        opt_d, opt_g = self.optimizers()

        # Train discriminator
        opt_d.zero_grad()
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
        d_loss.backward()
        opt_d.step()

        # Train generator
        opt_g.zero_grad()
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
        g_loss.backward()
        opt_g.step()

        # Log losses every 50 batches
        if batch_idx % 50 == 0:
            self.logger.experiment.log(
                {
                    "losses/d_real": real_loss,
                    "losses/d_fake": fake_loss,
                    "losses/d": d_loss,
                    "losses/g": g_loss,
                }
            )

        # Log generated images every 250 batches
        if batch_idx % 250 == 0:
            with torch.no_grad():
                img_grid_fake = torchvision.utils.make_grid(
                    fake_imgs[:16], normalize=True
                )
                img_grid_real = torchvision.utils.make_grid(
                    real_imgs[:16], normalize=True
                )
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
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def train_dataloader(self):
        return get_cifar10_dataloader(
            target_class=self.target_class, batch_size=self.batch_size, num_workers=8
        )[0]

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
    batch_size: int = Option(64),
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
    gan = GAN(batch_size=batch_size)

    # Start the training process
    trainer.fit(gan)

    # Finish the wandb run
    wandb.finish()

    logger.info("Finished training!")


if __name__ == "__main__":
    app()
