"""Task 3 comparison: DCGAN vs StyleGAN samples."""

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.metrics import save_metrics


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_stylegan_samples(sample_dir: Path = Path("results/stylegan_samples")) -> List[Path]:
    """Find valid image files in the StyleGAN samples directory."""
    if not sample_dir.is_dir():
        return []
    files = [
        p for p in sample_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def compare_dcgan_stylegan(
    sample_dir: Path = Path("results/stylegan_samples"),
    generator_path: Path = Path("results/gan_G.pt"),
    output_path: Path = Path("results/figures/gan_vs_stylegan.png"),
    seed: int = 42,
) -> bool:
    """Compare local DCGAN samples against downloaded StyleGAN demo samples."""
    stylegan_files = find_stylegan_samples(sample_dir)
    if len(stylegan_files) < 10:
        print(
            f"Found {len(stylegan_files)} StyleGAN sample(s) in {sample_dir}. "
            "At least 10 images are required for comparison.\n"
            "Action required: Download 10 sample images manually from "
            "https://thispersondoesnotexist.com or the NVlabs StyleGAN3 demo "
            f"and place them into '{sample_dir}/'."
        )
        return False

    if not generator_path.is_file():
        print(
            f"Trained DCGAN weights not found at {generator_path}.\n"
            "Action required: Run DCGAN training on Colab first ('python -m src.gan') "
            f"to generate {generator_path}."
        )
        return False

    import torch
    import torch.nn.functional as F
    from src.data import get_device
    from src.gan import Generator

    device = get_device()

    # 1. Load DCGAN Generator and generate 10 samples
    netG = Generator(latent_dim=100).to(device)
    checkpoint = torch.load(generator_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        netG.load_state_dict(checkpoint["model_state_dict"])
    else:
        netG.load_state_dict(checkpoint)
    netG.eval()

    torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(10, 100, 1, 1, device=device)
        dcgan_raw = netG(noise)  # shape: (10, 1, 28, 28), in [-1, 1]
        dcgan_norm = (dcgan_raw + 1.0) / 2.0
        dcgan_norm = torch.clamp(dcgan_norm, 0.0, 1.0)
        # Upscale 28x28 to 256x256 using bilinear interpolation for display only
        dcgan_256 = F.interpolate(
            dcgan_norm, size=(256, 256), mode="bilinear", align_corners=False
        ).cpu()

    # 2. Load 10 StyleGAN images and resize to 256x256
    selected_files = stylegan_files[:10]
    stylegan_images = []
    for file_path in selected_files:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            stylegan_images.append(np.array(img_rgb))

    # 3. Create 2x10 comparison figure
    fig, axes = plt.subplots(2, 10, figsize=(20, 4.5))

    for i in range(10):
        # Top row: DCGAN
        axes[0, i].imshow(dcgan_256[i, 0].numpy(), cmap="gray")
        axes[0, i].axis("off")

        # Bottom row: StyleGAN
        axes[1, i].imshow(stylegan_images[i])
        axes[1, i].axis("off")

    # Add row titles
    axes[0, 0].set_title("DCGAN (28x28, trained here)", loc="left", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("StyleGAN (pretrained, hosted demo)", loc="left", fontsize=11, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")

    # Record metrics
    save_metrics("stylegan_samples_used", len(selected_files))
    return True


if __name__ == "__main__":
    compare_dcgan_stylegan()
