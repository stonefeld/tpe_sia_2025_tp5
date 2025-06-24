import numpy as np
from sklearn.decomposition import PCA

from ej2.src.autoencoders import VariationalAutoencoder
from ej2.src.dataset import load_emoji_dataset
from ej2.src.plots import (
    plot_generated_emojis,
    plot_generated_emojis_for_paths,
    plot_interpolation_paths_clean,
    plot_latent_space_emojis,
    plot_loss_history,
)
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam


def main():
    dataset_path = "assets/emoji_dataset.pkl"
    dataset = load_emoji_dataset(dataset_path)

    names = dataset["names"]
    characters = dataset["characters"]
    images = np.array([dataset["emojis"][name] for name in names])
    image_size = images.shape[1]
    data = images.reshape(images.shape[0], -1)

    data = np.clip(data, 0, 1).astype(np.float32)

    input_size = image_size * image_size
    latent_size = 50

    vae = VariationalAutoencoder(
        input_dim=input_size,
        latent_dim=latent_size,
        hidden_layers=[512, 256],
        tita=sigmoid,
        tita_prime=sigmoid_prime,
        optimizer=Adam(learning_rate=0.0001),
    )
    loss_history = vae.train(data, epochs=3000, batch_size=32)
    plot_loss_history(loss_history, title="VAE Loss During Training")

    _, _, z, _, _ = vae.forward(data)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(z)

    print("\n=== Generando emojis desde una distribución aprendida ===")
    print(
        "En lugar de usar N(0,1), vamos a muestrear desde una distribución normal "
        "ajustada a la media y desviación estándar del espacio latente de los datos reales."
    )

    # Calcular la media y desviación estándar del espacio latente aprendido
    z_mean = np.mean(z, axis=0)
    z_std = np.std(z, axis=0)

    print(f"Media aprendida (promedio de {latent_size} dims): {np.mean(z_mean):.3f}")
    print(f"Desv. Est. aprendida (promedio de {latent_size} dims): {np.mean(z_std):.3f}")

    # Generar muestras desde la distribución aprendida
    z_samples = np.random.normal(loc=z_mean, scale=z_std, size=(10, latent_size))
    generated_latent_2d = pca.transform(z_samples)

    plot_latent_space_emojis(
        latent_2d,
        characters,
        generated_samples=generated_latent_2d,
        title="Espacio Latente con Muestras de Distribución Aprendida",
    )

    decoder_activations = vae.decoder.forward(z_samples)
    generated = decoder_activations[-1]
    plot_generated_emojis(generated, image_size)

    _, _, _, _, decoder_activations = vae.forward(data)
    reconstructed = decoder_activations[-1]
    plot_generated_emojis(data, image_size)
    plot_generated_emojis(reconstructed, image_size)

    print("\n=== Generando gráfico de múltiples caminos de interpolación ===")

    plot_interpolation_paths_clean(
        latent_2d,
        characters,
        title="Múltiples Caminos de Interpolación en el Espacio Latente",
    )

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
