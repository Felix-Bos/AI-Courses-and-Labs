from __future__ import annotations

import gzip
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_moons


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MNIST_RAW = ROOT / "data" / "MNIST" / "raw"


def squared_cost_matrix(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    x0_sq = np.sum(x0 * x0, axis=1, keepdims=True)
    x1_sq = np.sum(x1 * x1, axis=1, keepdims=True).T
    return x0_sq + x1_sq - 2.0 * x0 @ x1.T


def optimal_assignment(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    cost = squared_cost_matrix(x0, x1)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty_like(col_ind)
    order[row_ind] = col_ind
    return order


def vector_stats(u_ind: np.ndarray, u_ot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm_ind = np.linalg.norm(u_ind, axis=1)
    norm_ot = np.linalg.norm(u_ot, axis=1)
    cosine = np.sum(u_ind * u_ot, axis=1) / (norm_ind * norm_ot + 1e-8)
    return norm_ind, norm_ot, cosine


def generate_moons_ot_figure(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n = 80

    x0 = rng.normal(size=(n, 2))
    x1, _ = make_moons(n_samples=n, noise=0.06, random_state=seed)
    x1 = x1 * np.array([1.25, 1.15]) + np.array([0.0, 0.15])
    x1 = x1[rng.permutation(n)]

    independent_perm = rng.permutation(n)
    ot_perm = optimal_assignment(x0, x1)

    target_ind = x1[independent_perm]
    target_ot = x1[ot_perm]
    u_ind = target_ind - x0
    u_ot = target_ot - x0
    norm_ind, norm_ot, cosine = vector_stats(u_ind, u_ot)

    fig = plt.figure(figsize=(12.5, 4.2), dpi=170)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.85])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, target, title, color in [
        (axes[0], target_ind, "Couplage indépendant", "tab:red"),
        (axes[1], target_ot, "Couplage OT minibatch", "tab:green"),
    ]:
        ax.scatter(x0[:, 0], x0[:, 1], s=14, c="black", alpha=0.55, label=r"$x_0$")
        ax.scatter(x1[:, 0], x1[:, 1], s=14, c="tab:blue", alpha=0.55, label=r"$x_1$")
        step = 2
        ax.quiver(
            x0[::step, 0],
            x0[::step, 1],
            (target - x0)[::step, 0],
            (target - x0)[::step, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color=color,
            alpha=0.72,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlim(-3.0, 3.1)
        ax.set_ylim(-2.7, 2.7)

    axes[0].legend(loc="upper left", fontsize=8)
    axes[2].hist(norm_ind, bins=14, alpha=0.65, label="indépendant", color="tab:red")
    axes[2].hist(norm_ot, bins=14, alpha=0.65, label="OT", color="tab:green")
    axes[2].axvline(norm_ind.mean(), color="tab:red", linestyle="--", lw=1.2)
    axes[2].axvline(norm_ot.mean(), color="tab:green", linestyle="--", lw=1.2)
    axes[2].set_title("Normes des vecteurs")
    axes[2].set_xlabel(r"$\|x_1-x_0\|_2$")
    axes[2].set_ylabel("count")
    axes[2].legend(fontsize=8)
    axes[2].text(
        0.02,
        0.98,
        f"norme moy. ind. = {norm_ind.mean():.2f}\n"
        f"norme moy. OT = {norm_ot.mean():.2f}\n"
        f"cos(u_ind,u_OT) moy. = {cosine.mean():.2f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85"),
    )

    fig.suptitle("make_moons : comparaison des vecteurs de transport", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "ot_moons_vector_comparison.png", bbox_inches="tight")
    plt.close(fig)


def read_idx_images(path: Path, n_images: int) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX magic number: {magic}")
        count = min(count, n_images)
        data = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)
    return data.reshape(count, rows, cols).astype(np.float32) / 255.0


def generate_mnist_ot_figure(seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    image_path = MNIST_RAW / "train-images-idx3-ubyte"
    if not image_path.exists():
        image_path = MNIST_RAW / "train-images-idx3-ubyte.gz"

    images = read_idx_images(image_path, n_images=512)
    batch_size = 32
    ids = rng.choice(len(images), size=batch_size, replace=False)
    x1 = images[ids] * 2.0 - 1.0
    x1_flat = x1.reshape(batch_size, -1)

    x0 = rng.normal(size=x1_flat.shape).astype(np.float32)
    x0 = np.clip(x0 / 2.5, -1.0, 1.0)

    independent_perm = rng.permutation(batch_size)
    ot_perm = optimal_assignment(x0, x1_flat)

    target_ind = x1_flat[independent_perm]
    target_ot = x1_flat[ot_perm]
    u_ind = target_ind - x0
    u_ot = target_ot - x0
    norm_ind, norm_ot, cosine = vector_stats(u_ind, u_ot)

    chosen = np.argsort(norm_ind - norm_ot)[-6:][::-1]
    rows = ["bruit $x_0$", "cible ind.", "vecteur ind.", "cible OT", "vecteur OT", "milieu OT"]
    fig, axes = plt.subplots(len(rows), len(chosen), figsize=(10.5, 8.4), dpi=170)

    for col, idx in enumerate(chosen):
        noise = x0[idx].reshape(28, 28)
        ind_img = target_ind[idx].reshape(28, 28)
        ot_img = target_ot[idx].reshape(28, 28)
        vec_ind = u_ind[idx].reshape(28, 28)
        vec_ot = u_ot[idx].reshape(28, 28)
        mid_ot = 0.5 * noise + 0.5 * ot_img
        row_data = [noise, ind_img, vec_ind, ot_img, vec_ot, mid_ot]

        for row, arr in enumerate(row_data):
            ax = axes[row, col]
            if "vecteur" in rows[row]:
                vmax = np.percentile(np.abs(arr), 98)
                ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                ax.imshow((arr + 1.0) / 2.0, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"#{idx}\n"
                    f"||u_ind||={norm_ind[idx]:.1f}\n"
                    f"||u_OT||={norm_ot[idx]:.1f}",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=9)

    fig.suptitle(
        "MNIST : comparaison des vecteurs de transport en espace pixel",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(
        0.5,
        0.015,
        f"Batch size = {batch_size} | norme moyenne indépendante = {norm_ind.mean():.1f} | "
        f"norme moyenne OT = {norm_ot.mean():.1f} | cosinus moyen = {cosine.mean():.2f}",
        ha="center",
        fontsize=9,
    )
    fig.savefig(FIGURES / "ot_mnist_vector_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    generate_moons_ot_figure()
    generate_mnist_ot_figure()
    print("wrote", FIGURES / "ot_moons_vector_comparison.png")
    print("wrote", FIGURES / "ot_mnist_vector_comparison.png")
