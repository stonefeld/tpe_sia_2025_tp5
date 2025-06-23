import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from ej1.src.plots import plot_latent_space
from ej2.src.autoencoders import VariationalAutoencoder
from ej2.src.dataset import load_emoji_dataset
from ej2.src.plots import (
    plot_generated_emojis,
    plot_latent_space_emojis,
    plot_interpolation_paths_clean,
    plot_generated_emojis_for_paths,
)
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam
from shared.utils import pca_2d


def main():
    dataset_path = "assets/emoji_dataset.pkl"
    dataset = load_emoji_dataset(dataset_path)

    names = dataset["names"]
    characters = dataset["characters"]
    images = np.array([dataset["emojis"][name] for name in names])
    image_size = images.shape[1]
    data = images.reshape(images.shape[0], -1)

    # Ensure data is properly normalized to [0, 1]
    data = np.clip(data, 0, 1).astype(np.float32)

    input_size = image_size * image_size
    latent_size = 20

    # Create VAE with the new interface
    vae = VariationalAutoencoder(
        input_dim=input_size,
        latent_dim=latent_size,
        hidden_layers=[400, 300],
        tita=sigmoid,
        tita_prime=sigmoid_prime,
        optimizer=Adam(learning_rate=0.0001),
    )
    vae.train(data, epochs=3000, batch_size=32)

    _, _, z, _, _ = vae.forward(data)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(z)
    z_samples = np.random.normal(0, 1, size=(10, latent_size))
    generated_latent_2d = pca.transform(z_samples)

    plot_latent_space_emojis(latent_2d, characters, generated_samples=generated_latent_2d)

    decoder_activations = vae.decoder.forward(z_samples)
    generated = decoder_activations[-1]
    plot_generated_emojis(generated, image_size)

    _, _, _, _, decoder_activations = vae.forward(data)
    reconstructed = decoder_activations[-1]
    print(f"{reconstructed.shape=}, {data.shape=}")
    plot_generated_emojis(data, image_size)
    plot_generated_emojis(reconstructed, image_size)

    # Gráfico de múltiples caminos de interpolación
    print("\n=== Generando gráfico de múltiples caminos de interpolación ===")

    # Gráfico 1: Múltiples caminos en el espacio latente (versión limpia)
    plot_interpolation_paths_clean(
        latent_2d,
        characters,
        title="Múltiples Caminos de Interpolación en el Espacio Latente",
    )

    # Gráfico 2: Emojis generados para los caminos de interpolación
    print("\n=== Generando emojis para los caminos de interpolación ===")
    plot_generated_emojis_for_paths(
        z,
        vae,
        image_size,
        characters,
        title="Generación de Emojis para Diferentes Caminos en el Espacio Latente",
    )


if __name__ == "__main__":
    main()
