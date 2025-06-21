import sys

from ej1.src.autoencoders import Autoencoder
from ej1.src.dataset import FONT_DATA, decode_font
from ej1.src.plots import plot_all_letters, plot_error_distribution, plot_latent_space
from ej1.src.utils import build_params, pca_2d
from shared.utils import pixel_error


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} config.json")
        sys.exit(1)

    decoded_data = decode_font(FONT_DATA)
    plot_all_letters(decoded_data)

    init_params, train_params = build_params(sys.argv[1])
    autoencoder = Autoencoder(**init_params)
    autoencoder.train(decoded_data, **train_params)
    latent_representations = autoencoder.get_latent_representations(decoded_data)

    latent_2d = pca_2d(latent_representations)
    plot_latent_space(latent_2d, decoded_data)

    reconstructed = autoencoder.forward(decoded_data)[-1]
    errors = pixel_error(decoded_data, reconstructed)
    print("\n=============================\nPixel errors for each letter:")
    for i, err in enumerate(errors):
        print(f"Letter {i}: {int(err)} pixels")

    plot_all_letters(reconstructed)
    plot_error_distribution(errors)


if __name__ == "__main__":
    main()
