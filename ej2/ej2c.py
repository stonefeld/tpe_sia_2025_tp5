import numpy as np

from ej1.src.plots import plot_latent_space
from ej2.src.autoencoders import VariationalAutoencoder
from ej2.src.dataset import load_emoji_dataset
from ej2.src.plots import plot_generated_emojis
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam
from shared.utils import pca_2d


def main():
    dataset_path = "assets/emoji_dataset.pkl"
    dataset = load_emoji_dataset(dataset_path)

    names = dataset["names"]
    images = np.array([dataset["emojis"][name] for name in names])[:10]
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
        hidden_layers=[64, 32, 8],
        learning_rate=0.0001,
        tita=sigmoid,
        tita_prime=sigmoid_prime,
        optimizer=Adam(learning_rate=0.0001),
    )
    vae.train(data, epochs=5000, batch_size=32)

    encoder_activations = vae.encoder.forward(data)
    latent_representations = encoder_activations[-1]
    latent_2d = pca_2d(latent_representations)
    plot_latent_space(latent_2d, data)

    z_samples = np.random.normal(0, 1, size=(10, latent_size))
    decoder_activations = vae.decoder.forward(z_samples)
    generated = decoder_activations[-1]
    plot_generated_emojis(generated, image_size)

    x_hat, _, _, _ = vae.forward(data)
    plot_generated_emojis(x_hat, image_size)


if __name__ == "__main__":
    main()
