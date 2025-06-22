import numpy as np

from ej1.src.plots import plot_latent_space
from ej2.src.autoencoders import VariationalAutoencoder
from ej2.src.dataset import load_emoji_dataset
from ej2.src.plots import plot_generated_emojis, plot_latent_grid
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam
from shared.utils import pca_2d


def main():
    dataset_path = "assets/emoji_dataset.pkl"
    dataset = load_emoji_dataset(dataset_path)

    names = dataset["names"]
    images = np.array([dataset["emojis"][name] for name in names])
    image_size = images.shape[1]
    data = images.reshape(images.shape[0], -1)

    # Ensure data is properly normalized to [0, 1]
    data = np.clip(data, 0, 1).astype(np.float32)

    input_size = image_size * image_size
    latent_size = 2

    # Create VAE with the new interface
    vae = VariationalAutoencoder(
        input_dim=input_size,
        latent_dim=latent_size,
        hidden_layers=[50],
        tita=sigmoid,
        tita_prime=sigmoid_prime,
        optimizer=Adam(learning_rate=0.001),
    )
    vae.train(data, epochs=1000, batch_size=32)

    _, _, _, z = vae.forward(data)
    latent_2d = pca_2d(z)
    plot_latent_space(latent_2d, data)

    z_samples = np.random.normal(0, 1, size=(10, latent_size))
    decoder_activations = vae.decoder.forward(z_samples)
    generated = decoder_activations[-1]
    plot_generated_emojis(generated, image_size)

    reconstructed, _, _, _ = vae.forward(data)
    print(f"{reconstructed.shape=}, {data.shape=}")
    plot_generated_emojis(data, image_size)
    plot_generated_emojis(reconstructed, image_size)

    plot_latent_grid(vae, grid_size=10, range_z=2.5)


if __name__ == "__main__":
    main()
