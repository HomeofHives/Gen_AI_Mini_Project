"""Task 3: Deep Convolutional GAN (DCGAN) for Fashion-MNIST."""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data import get_device, get_fashion_mnist
from src.metrics import save_metrics


def weights_init(m: nn.Module) -> None:
    """DCGAN weight initialization: Conv ~ N(0, 0.02), BatchNorm ~ N(1, 0.02), bias 0."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """DCGAN Generator mapping 100-D latent noise vector to 1x28x28 image."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # (100, 1, 1) -> (128, 7, 7)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 7, 7) -> (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.main(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator classifying 1x28x28 image as real or fake."""

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 7, 7) -> (1, 1, 1)
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        return out.view(out.size(0), -1)


def compute_diversity_metrics(
    netG: Generator,
    test_dataset,
    device: torch.device,
    num_samples: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean pairwise L2 distance using torch.cdist for generated and real images."""
    torch.manual_seed(seed)
    netG.eval()

    # 1. 500 Generated samples
    with torch.no_grad():
        noise = torch.randn(num_samples, netG.latent_dim, 1, 1, device=device)
        fake_imgs = netG(noise).cpu()
    fake_flat = fake_imgs.view(num_samples, -1)

    # 2. 500 Real test samples
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=False)
    real_imgs, _ = next(iter(test_loader))
    # Scale real images to [-1, 1] to match Generator Tanh output range
    real_imgs_scaled = real_imgs * 2.0 - 1.0
    real_flat = real_imgs_scaled.view(num_samples, -1)

    # Mean pairwise L2 distance using torch.cdist
    # Upper-triangle non-diagonal indices for unique pairs
    triu_idx = torch.triu_indices(num_samples, num_samples, offset=1)

    dist_fake = torch.cdist(fake_flat, fake_flat, p=2)
    mean_dist_fake = float(dist_fake[triu_idx[0], triu_idx[1]].mean().item())

    dist_real = torch.cdist(real_flat, real_flat, p=2)
    mean_dist_real = float(dist_real[triu_idx[0], triu_idx[1]].mean().item())

    diversity_ratio = float(mean_dist_fake / mean_dist_real)

    save_metrics("gan_diversity_generated", mean_dist_fake)
    save_metrics("gan_diversity_real", mean_dist_real)
    save_metrics("gan_diversity_ratio", diversity_ratio)

    print(f"Mean Pairwise L2 Generated : {mean_dist_fake:.4f}")
    print(f"Mean Pairwise L2 Real      : {mean_dist_real:.4f}")
    print(f"Diversity Ratio (Gen/Real) : {diversity_ratio:.4f}")

    return diversity_ratio, mean_dist_fake, mean_dist_real


def train_dcgan(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 2e-4,
    beta1: float = 0.5,
    seed: int = 42,
) -> Tuple[Generator, Discriminator]:
    """Train DCGAN on Fashion-MNIST dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    train_dataset = get_fashion_mnist(train=True)
    test_dataset = get_fashion_mnist(train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    netG = Generator(latent_dim=100).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    epoch_dir = Path("results/figures/gan_epochs")
    epoch_dir.mkdir(parents=True, exist_ok=True)

    d_epoch_losses = []
    g_epoch_losses = []

    print(f"Starting DCGAN training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        netG.train()
        netD.train()
        total_d_loss = 0.0
        total_g_loss = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            b_size = real_imgs.size(0)
            real_imgs = (real_imgs * 2.0 - 1.0).to(device)

            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            # ---------------------
            #  1. Update Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            out_real = netD(real_imgs)
            loss_d_real = criterion(out_real, real_labels)

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_imgs = netG(noise)
            out_fake = netD(fake_imgs.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()

            # ---------------------
            #  2. Update Generator
            # ---------------------
            optimizer_G.zero_grad()
            out_g_fake = netD(fake_imgs)
            loss_g = criterion(out_g_fake, real_labels)
            loss_g.backward()
            optimizer_G.step()

            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        d_epoch_losses.append(avg_d_loss)
        g_epoch_losses.append(avg_g_loss)
        print(f"Epoch [{epoch:>2}/{epochs}] Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

        # Save fixed-noise 64-sample grid
        with torch.no_grad():
            sample_grid = netG(fixed_noise).cpu()
        grid_path = epoch_dir / f"epoch_{epoch:02d}.png"
        save_image(sample_grid, grid_path, nrow=8, normalize=True, value_range=(-1, 1))

    # 1. Save loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, epochs + 1), d_epoch_losses, label="Discriminator Loss", color="crimson")
    ax.plot(range(1, epochs + 1), g_epoch_losses, label="Generator Loss", color="royalblue")
    ax.set_title("DCGAN Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_path = Path("results/figures/gan_loss.png")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curves to {loss_path}")

    # 2. Save generator weights
    g_save_path = Path("results/gan_G.pt")
    g_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(netG.state_dict(), g_save_path)
    print(f"Saved Generator weights to {g_save_path}")

    # 3. Compute diversity ratio
    compute_diversity_metrics(netG, test_dataset, device, num_samples=500, seed=seed)

    return netG, netD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_dcgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
