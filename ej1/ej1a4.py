import os
import sys

import numpy as np

from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, decode_font
from ej1.src.utils import build_params, generate_from_latent
from ej1.src.plots import (
    plot_interpolated_images,
    plot_latent_space_with_interpolation,
    plot_complete_latent_space_visualization,
    plot_generated_images_for_paths,
)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} config.json")
        sys.exit(1)

    images = decode_font(FONT_DATA)

    init_params, train_params = build_params(sys.argv[1])
    model = Autoencoder(**init_params)
    model.train(images, **train_params)

    latent_repr = model.get_latent_representations(images)
    z_start = latent_repr[1]  # letra 'a'
    z_end = latent_repr[26]  # letra 'z'

    alphas = np.linspace(0, 1, 5)
    z_values = np.array([z_start * (1 - alpha) + z_end * alpha for alpha in alphas])

    generated = generate_from_latent(model, z_values)

    plot_interpolated_images(generated, alphas, "Imágenes Generadas por Interpolación en Espacio Latente (de 'a' a 'z')")
    plot_latent_space_with_interpolation(latent_repr, z_values, z_start, z_end)
    plot_complete_latent_space_visualization(latent_repr)
    plot_generated_images_for_paths(model, latent_repr, generate_from_latent)


if __name__ == "__main__":
    main()
