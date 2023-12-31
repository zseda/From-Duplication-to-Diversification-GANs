import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from loguru import logger
from pathlib import Path
import uuid
from src.data import get_cifar10_dataloader as get_dataloader
from src.models import DCDiscriminator, DCGenerator, weights_init_normal
from torchvision.utils import make_grid
import timm
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


def main(
    root_path: str = typer.Option("."),
    epochs: int = typer.Option(100),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    z_dim: int = typer.Option(100),
    start_c_after: int = typer.Option(15),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    # %80=48.000 %60=36.000 %40 =24.000 %20 = 12.000
    dataset_size: int = typer.Option(48000),
):
    current_time = datetime.now()
    session_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # Weights & Biases setup for online-only logging
    wandb.init(
        project="GAN-CIFAR10",
        name="DCGAN-train-" + session_name,
        settings=wandb.Settings(mode="online"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"batch_size: {batch_size}")
    tb_path = Path(root_path, "logs", experiment_id)
    tb_path.mkdir(parents=True, exist_ok=False)
    logger.info(f"experiment id: {experiment_id}")

    # load data
    loader_train = get_dataloader(batch_size=batch_size, num_workers=num_workers)[0]

    # classifier
    C = timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=10, in_chans=3
    )

    C.to(device)

    # initialize G
    G = DCGenerator(g_input_dim=z_dim)

    G.to(device)
    G.apply(weights_init_normal)

    # config D - weight norm
    D_weight_norm = nn.utils.spectral_norm

    # config D - norm
    D_norm = nn.Identity

    # config D - activation
    D_activation = nn.LeakyReLU

    # initialize D
    D = DCDiscriminator(norm=D_norm, weight_norm=D_weight_norm, activation=D_activation)

    D.to(device)
    D.apply(weights_init_normal)

    # optimizer
    C_optimizer = optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # loss
    criterion_classification = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    global_step = 0
    D_losses, G_losses = [], []

    # test labels
    labels_test = list()
    eye10 = torch.eye(10)
    for i in range(10):
        labels_test.append(eye10[i].repeat(8).view(8, 10))
    labels_test = torch.stack(labels_test).view(-1, 10).float().to(device)

    # persistent z for testing
    torch.manual_seed(42)
    z_permanent = torch.randn(80, z_dim).to(device)

    for e in range(1, epochs + 1):
        logger.info(f"training epoch {e}/{epochs}")
        for batch in loader_train:
            """
            -----------
            preparation
            -----------
            """
            # for logging
            global_step += 1

            # decompose batch data
            img, labels_real = batch
            img = img.to(device)
            labels_real = labels_real.to(device)
            labels_real_onehot = (
                F.one_hot(labels_real, num_classes=10).float().to(device)
            )

            actual_batch_size = img.shape[0]

            # reset gradients
            D.zero_grad()
            G.zero_grad()
            C.zero_grad()

            # create labels
            y_real = Variable(torch.ones(actual_batch_size, 1).to(device))
            y_fake = Variable(torch.zeros(actual_batch_size, 1).to(device))

            # generate random noise for G
            z = Variable(torch.randn(actual_batch_size, z_dim).to(device))

            # create class labels for generator - uniform distribution
            labels_fake = torch.randint(low=0, high=9, size=(actual_batch_size,))

            # one-hot encode class labels
            labels_fake_onehot = (
                F.one_hot(labels_fake, num_classes=10).float().to(device)
            )

            # generate images for D
            x_fake, x_fake_logits = G(z, labels_fake_onehot)

            """
                -------
                train C
                -------
            """
            C_out = C(img)
            C_loss = criterion_classification(C_out, labels_real_onehot)
            C_loss.backward()
            C_optimizer.step()

            """ 
                -------
                train D
                -------
            """
            # D_out_real = D(img, labels_real_onehot)
            D_out_real = D(img, labels_real_onehot)
            D_real_loss = criterion(D_out_real, y_real)

            D_out_fake = D(x_fake, labels_fake_onehot)
            D_fake_loss = criterion(D_out_fake, y_fake)

            # gradient backprop & optimize ONLY D's parameters
            D_loss = (D_real_loss + D_fake_loss) / 2.0
            D_loss.backward()
            D_optimizer.step()

            """ 
                -------
                train G
                -------
            """
            # reset gradients
            D.zero_grad()
            G.zero_grad()
            C.zero_grad()

            # generate images via G
            # create labels for testing generator
            # convert to one hot encoding
            z = Variable(torch.randn(actual_batch_size, z_dim).to(device))

            G_output, G_output_logits = G(z, labels_fake_onehot)
            D_out = D(G_output, labels_fake_onehot)
            G_disc_loss = criterion(D_out, y_real)

            # test generated images with classifier
            if e > start_c_after:
                C_out = C(G_output)
                G_classification_loss = criterion_classification(
                    C_out, labels_fake_onehot
                )
                G_loss = G_disc_loss + 0.3 * G_classification_loss
            else:
                G_loss = G_disc_loss

            # gradient backprop & optimize ONLY G's parameters
            G_loss.backward()
            G_optimizer.step()

            """
                -------
                logging
                -------
            """
            plot_img = (img + 1.0) / 2.0
            plot_output = (G_output + 1.0) / 2.0

            # print every 50 steps
            if global_step % 50 == 0:
                wandb.log(
                    {"train/disciriminator_loss": D_loss.item()}, step=global_step
                )
                wandb.log({"train/G_loss": G_loss.item()}, step=global_step)
                wandb.log({"train/G_disc_loss": G_disc_loss.item()}, step=global_step)
                if e > start_c_after:
                    wandb.log(
                        {"train/G_classification_loss": G_classification_loss.item()},
                        step=global_step,
                    )

            D_losses.append(D_loss.data.item())
            G_losses.append(G_loss.data.item())

            # Log images every 250 steps
            if global_step % 250 == 0:
                # Normalize images
                plot_img = (img + 1.0) / 2.0
                plot_output = (G_output + 1.0) / 2.0

                # Log images to WandB
                wandb.log(
                    {
                        "train/img": [wandb.Image(plot_img, caption="Image")],
                        "train/pred": [wandb.Image(plot_output, caption="Prediction")],
                    },
                    step=global_step,
                )

            """
            -------------------------------
            test class generation per epoch
            -------------------------------
        """
        with torch.no_grad():
            # Create random noise for G to generate images
            z = torch.randn(80, z_dim).to(device)
            # Make prediction - use labels test => structured one-hot encoded labels
            G_test_output, G_test_output_logits = G(z, labels_test)
            # Normalize images => output of G is tanh so we need to normalize to [0.0, 1.0]
            G_test_output = (G_test_output + 1.0) / 2.0
            # Save to WandB
            wandb.log(
                {
                    "test/pred": [
                        wandb.Image(make_grid(G_test_output), caption="Prediction")
                    ]
                },
                step=e,
            )

            # Make prediction of z_permanent - same random numbers for all epochs
            G_test_output_perm, G_test_output_logits = G(z_permanent, labels_test)
            # Normalize images => output of G is tanh so we need to normalize to [0.0, 1.0]
            G_test_output_perm = (G_test_output_perm + 1.0) / 2.0
            # Save to WandB
            wandb.log(
                {
                    "test/pred_perm": [
                        wandb.Image(
                            make_grid(G_test_output_perm),
                            caption="Permanent Prediction",
                        )
                    ]
                },
                step=e,
            )

        """
            ------
            saving
            ------
        """
        # save D
        if e % 5 == 0:
            torch.save(
                D.state_dict(),
                Path(
                    root_path, "logs", experiment_id, f"model_epoch_D{e:0>3}.pth"
                ).as_posix(),
            )

        # save G
        if e % 5 == 0:
            torch.save(
                G.state_dict(),
                Path(
                    root_path, "logs", experiment_id, f"model_epoch_G{e:0>3}.pth"
                ).as_posix(),
            )


if __name__ == "__main__":
    typer.run(main)
    wandb.finish()
