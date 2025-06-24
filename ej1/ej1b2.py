import numpy as np

from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, add_salt_and_pepper_noise, decode_font
from ej1.src.plots import plot_all_letters, plot_noise_level_comparison, plot_noise_reconstruction_comparison
from shared.activators import sigmoid, sigmoid_prime
from shared.metrics import compute_mse
from shared.optimizers import Adam
from shared.utils import pixel_error


def main():
    layers = [35, 64, 32, 8, 32, 64, 35]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {"noise": [], "MSE": [], "Error promedio (Píxeles)": [], "Error máximo (Píxeles)": []}
    images = decode_font(FONT_DATA)

    all_noisy_images = []
    all_reconstructed_images = []

    epochs = 3000
    batch_size = 8
    learning_rate = 0.01

    for noise in noise_levels:
        print(f"\nEvaluando con ruido = {noise}")

        optimizer = Adam(learning_rate=learning_rate, layers=layers)
        model = Autoencoder(layers=layers, tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)

        noisy_images = add_salt_and_pepper_noise(images, noise)
        plot_all_letters(noisy_images)
        model.train(images, epochs=epochs, batch_size=batch_size, max_pixel_error=None)

        reconstructed = model.forward(noisy_images)[-1]
        errors = pixel_error(images, reconstructed)

        mse = compute_mse(images, reconstructed)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        results["noise"].append(noise)
        results["MSE"].append(mse)
        results["Error promedio (Píxeles)"].append(mean_error)
        results["Error máximo (Píxeles)"].append(max_error)

        all_noisy_images.append(noisy_images)
        all_reconstructed_images.append(reconstructed)

    plot_noise_reconstruction_comparison(all_noisy_images, all_reconstructed_images, noise_levels, threshold=True)
    plot_noise_level_comparison(results, layers, epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
