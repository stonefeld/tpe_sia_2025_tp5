import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from shared.utils import save_plot


def plot_generated_emojis(generated_images, image_size):
    n_images = len(generated_images)
    n_cols = 8
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axs = axs.flatten()

    for i in range(n_images):
        axs[i].imshow(generated_images[i].reshape(image_size, image_size), cmap="gray")
        axs[i].axis("off")

    for i in range(n_images, len(axs)):
        axs[i].axis("off")

    fig.suptitle("Emojis generados")
    plt.tight_layout()
    save_plot(fig, "results/generated_emojis.png")
    plt.show()


def plot_latent_space(vae, data, labels=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, x in enumerate(data):
        mu, _ = vae.encode(x.reshape(1, -1))  # (1, input_dim)
        z = mu.flatten()  # (2,)

        # Obtener reconstrucción
        recon = vae.decode(z.reshape(1, -1))  # (1, input_dim)
        emoji = recon.reshape(7, 5)  # 5x7

        # Mostrar como imagen embebida
        img = OffsetImage(emoji, cmap="gray", zoom=4)
        ab = AnnotationBbox(img, (z[0], z[1]), frameon=False)
        ax.add_artist(ab)

        # Si querés etiquetas
        if labels is not None:
            ax.text(z[0], z[1], str(labels[i]), fontsize=8)

    ax.set_title("Espacio Latente (mu) con emojis")
    ax.grid(True)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()


def plot_latent_grid(vae, grid_size=10, range_z=2.5):
    digit_size = (7, 5)  # Tamaño de cada emoji
    figure = np.zeros((digit_size[0] * grid_size, digit_size[1] * grid_size))

    # Coordenadas en z
    grid_x = np.linspace(-range_z, range_z, grid_size)
    grid_y = np.linspace(-range_z, range_z, grid_size)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decode(z_sample)
            emoji = x_decoded.reshape(digit_size)

            top = i * digit_size[0]
            left = j * digit_size[1]
            figure[top : top + digit_size[0], left : left + digit_size[1]] = emoji

    plt.figure(figsize=(8, 8))
    plt.imshow(figure, cmap="gray")
    plt.title("Mapa de generación en el espacio latente")
    plt.axis("off")
    plt.show()
