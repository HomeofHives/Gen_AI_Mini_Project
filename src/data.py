"""Data handling and device setup module for genai_ad_studio."""

import torch
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """Select compute device in order: CUDA -> MPS -> CPU.

    Returns:
        torch.device: Chosen PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device selected: {device}")
    return device


def get_fashion_mnist(train: bool = True, data_dir: str = "data"):
    """Download and load Fashion-MNIST dataset with ToTensor transform.

    Args:
        train (bool): True for training split, False for test split.
        data_dir (str): Root directory for data storage.

    Returns:
        torchvision.datasets.FashionMNIST: The loaded dataset.
    """
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    return dataset


if __name__ == "__main__":
    device = get_device()
    print("Testing Fashion-MNIST dataset loader...")
    train_data = get_fashion_mnist(train=True)
    print(f"Fashion-MNIST training samples loaded: {len(train_data)}")
