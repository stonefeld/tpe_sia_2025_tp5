import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10
import matplotlib.pyplot as plt

def compute_mse(original, reconstructed):
    """Error cuadrático medio entre original y reconstrucción"""
    return F.mse_loss(reconstructed, original).item()

def compute_psnr(original, reconstructed):
    """PSNR: relación señal-ruido pico entre imágenes"""
    mse = F.mse_loss(reconstructed, original).item()
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0 / np.sqrt(mse))

def compute_ssim(original, reconstructed):
    """
    SSIM: Índice de similitud estructural (solo imágenes 2D escala de grises)
    Se espera que el tensor tenga forma [batch, 1, H, W]
    """
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
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
    original_bin = (original > threshold).float()
    reconstructed_bin = (reconstructed > threshold).float()
    correct_pixels = (original_bin == reconstructed_bin).float().mean().item()
    return correct_pixels

def plot_reconstruction(original, noisy, reconstructed, n=5):
    """
    Grilla de n ejemplos: original | con ruido | reconstruido
    """
    fig, axs = plt.subplots(n, 3, figsize=(6, n*2))
    for i in range(n):
        axs[i, 0].imshow(original[i, 0].detach().cpu(), cmap='gray')
        axs[i, 0].set_title("Original")
        axs[i, 1].imshow(noisy[i, 0].detach().cpu(), cmap='gray')
        axs[i, 1].set_title("Noisy")
        axs[i, 2].imshow(reconstructed[i, 0].detach().cpu(), cmap='gray')
        axs[i, 2].set_title("Reconstructed")
        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
