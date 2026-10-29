"""Task 1: Statistical distribution sampling and analysis."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.metrics import save_metrics


def sample_distributions(
    n_samples: int = 5000, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate random samples from standard distributions.

    Args:
        n_samples (int): Number of samples per distribution.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Mapping of distribution name to numpy sample array.
    """
    rng = np.random.default_rng(seed)
    return {
        "gaussian": rng.normal(loc=0.0, scale=1.0, size=n_samples),
        "uniform": rng.uniform(low=-3.0, high=3.0, size=n_samples),
        "exponential": rng.exponential(scale=1.0, size=n_samples),
    }


def compute_distribution_stats(samples: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a 1D sample array.

    Args:
        samples (np.ndarray): 1D array of numeric samples.

    Returns:
        dict: mean, variance, skewness, and excess kurtosis.
    """
    mean_val = float(np.mean(samples))
    var_val = float(np.var(samples, ddof=1))
    skew_val = float(stats.skew(samples))
    kurt_val = float(stats.kurtosis(samples, fisher=True))

    return {
        "mean": mean_val,
        "variance": var_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
    }


def plot_distributions(
    samples_dict: Dict[str, np.ndarray],
    out_path: Path = Path("results/figures/distributions.png"),
) -> None:
    """Generate and save 2x3 visualization of PDFs and ECDFs.

    Args:
        samples_dict (dict): Dictionary of samples.
        out_path (Path): Filepath to save generated figure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)

    # 1. Gaussian
    g_samples = samples_dict["gaussian"]
    ax_hist_g, ax_cdf_g = axes[0, 0], axes[1, 0]
    ax_hist_g.hist(g_samples, bins=50, density=True, alpha=0.6, color="royalblue", edgecolor="white", label="Sample Hist")
    gx = np.linspace(-4.0, 4.0, 400)
    ax_hist_g.plot(gx, stats.norm.pdf(gx, 0, 1), "r-", lw=2, label="Theoretical PDF")
    ax_hist_g.set_title("Gaussian (mean=0, std=1)")
    ax_hist_g.set_ylabel("Density")
    ax_hist_g.legend()
    ax_hist_g.grid(True, alpha=0.3)

    g_sorted = np.sort(g_samples)
    g_cdf = np.arange(1, len(g_sorted) + 1) / len(g_sorted)
    ax_cdf_g.plot(g_sorted, g_cdf, color="royalblue", lw=2, label="Empirical CDF")
    ax_cdf_g.plot(gx, stats.norm.cdf(gx, 0, 1), "r--", lw=1.5, label="Theoretical CDF")
    ax_cdf_g.set_title("Gaussian ECDF")
    ax_cdf_g.set_xlabel("Value")
    ax_cdf_g.set_ylabel("Cumulative Prob")
    ax_cdf_g.legend()
    ax_cdf_g.grid(True, alpha=0.3)

    # 2. Uniform
    u_samples = samples_dict["uniform"]
    ax_hist_u, ax_cdf_u = axes[0, 1], axes[1, 1]
    ax_hist_u.hist(u_samples, bins=50, density=True, alpha=0.6, color="darkorange", edgecolor="white", label="Sample Hist")
    ux = np.linspace(-3.5, 3.5, 400)
    ax_hist_u.plot(ux, stats.uniform.pdf(ux, -3, 6), "r-", lw=2, label="Theoretical PDF")
    ax_hist_u.set_title("Uniform ([-3, 3])")
    ax_hist_u.legend()
    ax_hist_u.grid(True, alpha=0.3)

    u_sorted = np.sort(u_samples)
    u_cdf = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
    ax_cdf_u.plot(u_sorted, u_cdf, color="darkorange", lw=2, label="Empirical CDF")
    ax_cdf_u.plot(ux, stats.uniform.cdf(ux, -3, 6), "r--", lw=1.5, label="Theoretical CDF")
    ax_cdf_u.set_title("Uniform ECDF")
    ax_cdf_u.set_xlabel("Value")
    ax_cdf_u.legend()
    ax_cdf_u.grid(True, alpha=0.3)

    # 3. Exponential
    e_samples = samples_dict["exponential"]
    ax_hist_e, ax_cdf_e = axes[0, 2], axes[1, 2]
    ax_hist_e.hist(e_samples, bins=50, density=True, alpha=0.6, color="forestgreen", edgecolor="white", label="Sample Hist")
    ex = np.linspace(0.0, 8.0, 400)
    ax_hist_e.plot(ex, stats.expon.pdf(ex, scale=1.0), "r-", lw=2, label="Theoretical PDF")
    ax_hist_e.set_title("Exponential (rate=1.0)")
    ax_hist_e.legend()
    ax_hist_e.grid(True, alpha=0.3)

    e_sorted = np.sort(e_samples)
    e_cdf = np.arange(1, len(e_sorted) + 1) / len(e_sorted)
    ax_cdf_e.plot(e_sorted, e_cdf, color="forestgreen", lw=2, label="Empirical CDF")
    ax_cdf_e.plot(ex, stats.expon.cdf(ex, scale=1.0), "r--", lw=1.5, label="Theoretical CDF")
    ax_cdf_e.set_title("Exponential ECDF")
    ax_cdf_e.set_xlabel("Value")
    ax_cdf_e.legend()
    ax_cdf_e.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution figure to {out_path}")


def run_distributions_task() -> Dict[str, Dict[str, float]]:
    """Execute Task 1: sample, compute statistics, plot, and record metrics."""
    samples = sample_distributions(n_samples=5000, seed=42)

    all_stats: Dict[str, Dict[str, float]] = {}
    for name, data in samples.items():
        all_stats[name] = compute_distribution_stats(data)

    save_metrics("distributions", all_stats)

    plot_distributions(samples)

    # Print small table
    print("\n" + "=" * 65)
    print(f"{'Distribution':<15} {'Mean':>10} {'Variance':>10} {'Skewness':>12} {'Excess Kurt':>14}")
    print("-" * 65)
    for name, s in all_stats.items():
        print(f"{name.capitalize():<15} {s['mean']:>10.4f} {s['variance']:>10.4f} {s['skewness']:>12.4f} {s['excess_kurtosis']:>14.4f}")
    print("=" * 65 + "\n")

    return all_stats


if __name__ == "__main__":
    run_distributions_task()
