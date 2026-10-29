"""Task 2: Convolutional Variational Autoencoder (VAE) for Fashion-MNIST."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import get_device, get_fashion_mnist
from src.metrics import save_metrics


FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> Flatten -> fc_mu, fc_logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder: latent_dim -> 64*7*7 -> 32x14x14 -> 1x28x28 -> Sigmoid
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into Gaussian latent parameters (mu, logvar)."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z back to pixel space."""
        h = self.decoder_fc(z)
        return self.decoder_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning reconstruction, mu, and logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: BCE summed over pixels + KL, averaged over batch."""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + kl_div
    return total_loss, recon_loss, kl_div


def evaluate_and_visualize(
    model: ConvVAE,
    test_dataset,
    device: torch.device,
    output_dir: Path = Path("results/figures"),
) -> float:
    """Evaluate VAE test MSE and generate visualization figures."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            recon = model.decode(mu)
            mse_batch = F.mse_loss(recon, x, reduction="sum").item()
            total_mse += mse_batch
            total_samples += x.size(0) * 28 * 28

    mean_pixel_mse = float(total_mse / total_samples)
    save_metrics("vae_test_mse", mean_pixel_mse)
    print(f"Test per-pixel MSE: {mean_pixel_mse:.6f}")

    # 1. 10 Reconstructions (10 test images top, reconstructions below)
    test_sample_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    sample_imgs, _ = next(iter(test_sample_loader))
    with torch.no_grad():
        sample_imgs_dev = sample_imgs.to(device)
        sample_mu, _ = model.encode(sample_imgs_dev)
        sample_recons = model.decode(sample_mu).cpu()

    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    for i in range(10):
        axes[0, i].imshow(sample_imgs[i, 0].numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(sample_recons[i, 0].numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstructed", loc="left", fontsize=10)
    recon_path = output_dir / "vae_reconstructions.png"
    plt.tight_layout()
    plt.savefig(recon_path, dpi=150)
    plt.close(fig)
    print(f"Saved reconstructions to {recon_path}")

    # 2. Latent PCA (3,000 test images)
    pca_loader = DataLoader(test_dataset, batch_size=3000, shuffle=False)
    pca_imgs, pca_labels = next(iter(pca_loader))
    with torch.no_grad():
        pca_mu, _ = model.encode(pca_imgs.to(device))
        pca_mu_np = pca_mu.cpu().numpy()

    pca_labels_np = pca_labels.numpy()
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(pca_mu_np)
    except ImportError:
        # Fallback to SVD projection
        mu_centered = pca_mu_np - np.mean(pca_mu_np, axis=0)
        _, _, vh = np.linalg.svd(mu_centered, full_matrices=False)
        mu_2d = mu_centered @ vh[:2].T

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", 10)
    for class_idx in range(10):
        mask = pca_labels_np == class_idx
        ax.scatter(
            mu_2d[mask, 0],
            mu_2d[mask, 1],
            c=[cmap(class_idx)],
            label=FASHION_CLASSES[class_idx],
            alpha=0.6,
            s=12,
        )
    ax.set_title("VAE Latent Space 2D PCA (3,000 Test Samples)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    pca_path = output_dir / "vae_latent_pca.png"
    plt.savefig(pca_path, dpi=150)
    plt.close(fig)
    print(f"Saved latent PCA scatter to {pca_path}")

    # 3. Latent Interpolation (10 steps between test image 0 and 1)
    img0, _ = test_dataset[0]
    img1, _ = test_dataset[1]
    with torch.no_grad():
        mu0, _ = model.encode(img0.unsqueeze(0).to(device))
        mu1, _ = model.encode(img1.unsqueeze(0).to(device))
        alphas = np.linspace(0.0, 1.0, 10)
        interp_mu = torch.stack([(1 - a) * mu0 + a * mu1 for a in alphas]).squeeze(1)
        interp_imgs = model.decode(interp_mu).cpu()

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(interp_imgs[i, 0].numpy(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"a={alphas[i]:.2f}", fontsize=8)
    plt.tight_layout()
    interp_path = output_dir / "vae_interpolation.png"
    plt.savefig(interp_path, dpi=150)
    plt.close(fig)
    print(f"Saved latent interpolation to {interp_path}")

    return mean_pixel_mse


def train_vae(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 16,
    seed: int = 42,
) -> ConvVAE:
    """Train ConvVAE on Fashion-MNIST dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    train_dataset = get_fashion_mnist(train=True)
    test_dataset = get_fashion_mnist(train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting VAE training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, _, _ = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch:>2}/{epochs}] Average Loss: {avg_loss:.4f}")

    # Save model weights and latent dimension
    model_save_path = Path("results/vae_fashion.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
        },
        model_save_path,
    )
    print(f"Saved trained VAE model to {model_save_path}")

    evaluate_and_visualize(model, test_dataset, device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvVAE on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_vae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
