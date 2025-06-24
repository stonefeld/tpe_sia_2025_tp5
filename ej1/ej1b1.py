import numpy as np

from shared.activators import sigmoid, sigmoid_prime
from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, decode_font
from shared.optimizers import Adam
from ej1.src.plots import plot_architecture_comparison
from shared.metrics import compute_mse
from shared.utils import pixel_error


def main():
    architectures = [
        [35, 16, 2, 16, 35],
        [35, 32, 2, 32, 35],
        [35, 32, 16, 2, 16, 32, 35],
        [35, 64, 32, 4, 32, 64, 35],
        [35, 64, 32, 8, 32, 64, 35],
    ]

    results = {"arch": [], "MSE": [], "Error promedio (Píxeles)": [], "Error máximo (Píxeles)": []}
    images = decode_font(FONT_DATA)

    epochs = 3000
    batch_size = 8
    learning_rate = 0.01

    for arch in architectures:
        print(f"\nEntrenando arquitectura: {arch}")

        optimizer = Adam(learning_rate=learning_rate, layers=arch)
        model = Autoencoder(layers=arch, tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)
        model.train(images, epochs=epochs, batch_size=batch_size, max_pixel_error=None)

        reconstructed = model.forward(images)[-1]
        errors = pixel_error(images, reconstructed)

        mse = compute_mse(images, reconstructed)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        results["arch"].append(str(arch))
        results["MSE"].append(mse)
        results["Error promedio (Píxeles)"].append(mean_error)
        results["Error máximo (Píxeles)"].append(max_error)

    plot_architecture_comparison(results)


if __name__ == "__main__":
    main()
