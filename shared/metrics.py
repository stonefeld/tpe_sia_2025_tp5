from math import log10

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_mse(original, reconstructed):
    """MSE: error cuadrático medio entre imágenes"""
    original = np.array(original)
    reconstructed = np.array(reconstructed)
    return np.mean((reconstructed - original) ** 2)


def compute_psnr(original, reconstructed):
    """PSNR: relación señal-ruido pico entre imágenes"""
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    return 20 * log10(1.0 / np.sqrt(mse))


def compute_ssim(original, reconstructed):
    """
    SSIM: Índice de similitud estructural (solo imágenes 2D escala de grises)
    Se espera que el array tenga forma [batch, 1, H, W]
    """
    orig_np = np.array(original)
    recon_np = np.array(reconstructed)

    if orig_np.ndim == 4:
        scores = []
        for i in range(orig_np.shape[0]):
            scores.append(ssim(orig_np[i, 0], recon_np[i, 0]))
        return np.mean(scores)
    else:
        return ssim(orig_np, recon_np)


def compute_pixel_accuracy(original, reconstructed, threshold=0.5):
    """
    Exactitud de píxeles (para imágenes binarias). Compara pixel a pixel.
    """
    orig_np = np.array(original)
    recon_np = np.array(reconstructed)
    original_bin = (orig_np > threshold).astype(float)
    reconstructed_bin = (recon_np > threshold).astype(float)
    correct_pixels = np.mean(original_bin == reconstructed_bin)
    return correct_pixels


def plot_reconstruction(original, noisy, reconstructed, n=5):
    """
    Grilla de n ejemplos: original | con ruido | reconstruido
    """
    fig, axs = plt.subplots(n, 3, figsize=(6, n * 2))
    for i in range(n):
        axs[i, 0].imshow(np.array(original)[i, 0], cmap="gray")
        axs[i, 0].set_title("Original")
        axs[i, 1].imshow(np.array(noisy)[i, 0], cmap="gray")
        axs[i, 1].set_title("Noisy")
        axs[i, 2].imshow(np.array(reconstructed)[i, 0], cmap="gray")
        axs[i, 2].set_title("Reconstructed")
        for ax in axs[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.show()
