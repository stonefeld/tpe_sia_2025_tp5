import os
import sys

import numpy as np

from ej1.src.activators import sigmoid, sigmoid_prime
from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, decode_font
from ej1.src.optimizers import Adam
from ej1.src.plots import plot_architecture_comparison
from ej1.src.utils import compute_all_metrics, print_metrics_summary
from shared.utils import pixel_error


def main():
    architectures = [
        {"layers": [35, 16, 2, 16, 35]},
        {"layers": [35, 32, 2, 32, 35]},
        {"layers": [35, 32, 16, 2, 16, 32, 35]},
        {"layers": [35, 64, 32, 4, 32, 64, 35]},
        {"layers": [35, 64, 32, 8, 32, 64, 35]},
    ]

    results = {"arch": [], "MSE": [], "Pixel_Error": [], "Max_Error": [], "PSNR": [], "SSIM": [], "Pixel_Accuracy": []}
    images = decode_font(FONT_DATA)

    for arch in architectures:
        print(f"\nEntrenando arquitectura: {arch['layers']}")

        optimizer = Adam(learning_rate=0.001, layers=arch)
        model = Autoencoder(layers=arch["layers"], tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)
        model.train(images, epochs=1000, batch_size=8, max_pixel_error=None)

        reconstructed = model.forward(images)[-1]
        errors = pixel_error(images, reconstructed)

        metrics = compute_all_metrics(images, reconstructed)

        max_error = np.max(errors)
        mean_error = np.mean(errors)

        results["arch"].append(str(arch["layers"]))
        results["MSE"].append(metrics["MSE"])
        results["PSNR"].append(metrics["PSNR"])
        results["SSIM"].append(metrics["SSIM"])
        results["Pixel_Accuracy"].append(metrics["Pixel_Accuracy"])
        results["Pixel_Error"].append(mean_error)
        results["Max_Error"].append(max_error)

        print(f"MSE: {metrics['MSE']:.6f}, PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.6f}")
        print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f}, Mean Pixel Error: {mean_error:.2f}, Max Pixel Error: {max_error}")

    plot_architecture_comparison(results)
    print_metrics_summary(results)


if __name__ == "__main__":
    main()
