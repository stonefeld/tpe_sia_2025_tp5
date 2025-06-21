import numpy as np

from ej1.src.activators import sigmoid, sigmoid_prime
from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, decode_font
from ej1.src.optimizers import Adam
from ej1.src.plots import plot_noise_level_comparison
from shared.metrics import compute_mse
from shared.utils import pixel_error


def main():
    layers = [35, 32, 16, 2, 16, 32, 35]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = {"noise": [], "MSE": [], "Error promedio (Píxeles)": [], "Error máximo (Píxeles)": []}
    images = decode_font(FONT_DATA)

    for noise in noise_levels:
        print(f"\nEvaluando con ruido = {noise}")

        optimizer = Adam(learning_rate=0.001, layers=layers)
        model = Autoencoder(layers=layers, tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)
        model.train(images, epochs=500, batch_size=8, max_pixel_error=None)

        reconstructed = model.forward(images)[-1]
        errors = pixel_error(images, reconstructed)

        mse = compute_mse(images, reconstructed)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        results["noise"].append(noise)
        results["MSE"].append(mse)
        results["Error promedio (Píxeles)"].append(mean_error)
        results["Error máximo (Píxeles)"].append(max_error)

    plot_noise_level_comparison(results, layers, 500, 8, 0.001)


if __name__ == "__main__":
    main()
