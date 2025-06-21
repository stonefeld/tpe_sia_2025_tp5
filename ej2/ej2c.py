import matplotlib.pyplot as plt
import numpy as np

from ej1.src.autoencoders import Autoencoder
from ej2.src.autoencoders import VariationalAutoencoder
from ej2.src.dataset import load_emoji_dataset
from ej2.src.plots import plot_generated_emojis
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam


def main():
    dataset_path = "assets/emoji_dataset.pkl"
    dataset = load_emoji_dataset(dataset_path)

    names = dataset["names"]
    images = np.array([dataset["emojis"][name] for name in names])[10:20]
    image_size = images.shape[1]
    data = images.reshape(images.shape[0], -1)

    input_size = image_size * image_size
    hidden_size = 64
    latent_size = 2

    encoder_layers = [input_size, hidden_size, latent_size * 2]
    encoder = Autoencoder(encoder_layers, sigmoid, sigmoid_prime, Adam(learning_rate=0.001))

    decoder_layers = [latent_size, hidden_size, input_size]
    decoder = Autoencoder(decoder_layers, sigmoid, sigmoid_prime, Adam(learning_rate=0.001))

    vae = VariationalAutoencoder(encoder, decoder, Adam(learning_rate=0.0001), latent_size)

    train_data = data.astype(np.float32)
    epochs = 10000
    print("\nStarting VAE training...")

    for epoch in range(epochs):
        loss = vae.train_batch(train_data)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("Training complete.\n")
    print("Generating new emojis from latent space...")

    z_samples = np.random.normal(0, 1, size=(10, latent_size))
    decoder_activations = decoder.forward(z_samples)
    generated = decoder_activations[-1]
    plot_generated_emojis(generated, image_size)


if __name__ == "__main__":
    main()
