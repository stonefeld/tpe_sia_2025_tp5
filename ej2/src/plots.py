import matplotlib.pyplot as plt
import numpy as np

from shared.utils import save_plot


def plot_generated_emojis(generated_images, image_size, title="Generated Emojis"):
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

    fig.suptitle(title)
    plt.tight_layout()
    save_plot(fig, "results/generated_emojis.png")
    plt.show()


# TODO: si no se usa volarlo
def plot_interpolated_emojis(generated_images, alphas, image_size, title="Emojis Generados por Interpolación en Espacio Latente"):
    fig, axs = plt.subplots(1, len(generated_images), figsize=(15, 4))
    fig.suptitle(title)

    for i in range(len(generated_images)):
        generated_img = generated_images[i].reshape(image_size, image_size)
        axs[i].imshow(generated_img, cmap="gray", vmin=0, vmax=1)
        axs[i].set_title(f"Paso de interpolación {i + 1}\nα = {alphas[i]:.2f}")
        axs[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "results/interpolated_emojis.png")
    plt.show()


# TODO: si no se usa volarlo
def plot_multiple_interpolation_paths(
    latent_repr_2d, emoji_chars, vae, image_size, z_original=None, title="Múltiples Caminos de Interpolación"
):
    fig, ax = plt.subplots(figsize=(15, 12))

    ax.scatter(
        latent_repr_2d[:, 0],
        latent_repr_2d[:, 1],
        c="lightblue",
        s=120,
        alpha=0.6,
        label="Emojis Reales",
    )
    for i, (x, y) in enumerate(latent_repr_2d):
        ax.annotate(
            emoji_chars[i],
            (x, y),
            xytext=(7, -2),
            textcoords="offset points",
            fontsize=14,
        )

    caminos = [
        (0, -1, "red", f"{emoji_chars[0]} → {emoji_chars[-1]}"),
        (1, -2, "green", f"{emoji_chars[1]} → {emoji_chars[-2]}"),
        (2, -3, "blue", f"{emoji_chars[2]} → {emoji_chars[-3]}"),
        (3, -4, "orange", f"{emoji_chars[3]} → {emoji_chars[-4]}"),
        (4, -5, "purple", f"{emoji_chars[4]} → {emoji_chars[-5]}"),
    ]

    alphas = np.linspace(0, 1, 10)

    for inicio_idx, fin_idx, color, label in caminos:
        z_inicio_2d = latent_repr_2d[inicio_idx]
        z_fin_2d = latent_repr_2d[fin_idx]
        z_camino_2d = np.array([z_inicio_2d * (1 - alpha) + z_fin_2d * alpha for alpha in alphas])

        ax.plot(
            z_camino_2d[:, 0],
            z_camino_2d[:, 1],
            c=color,
            linewidth=2.5,
            alpha=0.8,
            label=label,
        )
        ax.scatter(z_camino_2d[:, 0], z_camino_2d[:, 1], c=color, s=50, alpha=0.9)

        ax.scatter(
            z_inicio_2d[0],
            z_inicio_2d[1],
            c=color,
            s=250,
            marker="o",
            edgecolors="black",
            linewidth=2,
        )
        ax.scatter(
            z_fin_2d[0],
            z_fin_2d[1],
            c=color,
            s=250,
            marker="s",
            edgecolors="black",
            linewidth=2,
        )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Dimensión Latente 1", fontsize=12)
    ax.set_ylabel("Dimensión Latente 2", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    save_plot(fig, "results/multiple_interpolation_paths.png")
    plt.show()


def plot_interpolation_paths_clean(latent_repr_2d, emoji_chars, title="Múltiples Caminos de Interpolación"):
    fig, ax = plt.subplots(figsize=(15, 12))

    ax.scatter(
        latent_repr_2d[:, 0],
        latent_repr_2d[:, 1],
        c="lightblue",
        s=120,
        alpha=0.6,
        label="Emojis Reales",
    )
    for i, (x, y) in enumerate(latent_repr_2d):
        ax.annotate(
            emoji_chars[i],
            (x, y),
            xytext=(7, -2),
            textcoords="offset points",
            fontsize=14,
        )

    caminos = [
        (0, -1, "red", f"{emoji_chars[0]} → {emoji_chars[-1]}"),
        (1, -2, "green", f"{emoji_chars[1]} → {emoji_chars[-2]}"),
        (2, -3, "blue", f"{emoji_chars[2]} → {emoji_chars[-3]}"),
        (3, -4, "orange", f"{emoji_chars[3]} → {emoji_chars[-4]}"),
        (4, -5, "purple", f"{emoji_chars[4]} → {emoji_chars[-5]}"),
    ]

    alphas = np.linspace(0, 1, 10)

    for inicio_idx, fin_idx, color, label in caminos:
        z_inicio_2d = latent_repr_2d[inicio_idx]
        z_fin_2d = latent_repr_2d[fin_idx]
        z_camino_2d = np.array([z_inicio_2d * (1 - alpha) + z_fin_2d * alpha for alpha in alphas])

        ax.plot(
            z_camino_2d[:, 0],
            z_camino_2d[:, 1],
            c=color,
            linewidth=2.5,
            alpha=0.8,
            label=label,
        )
        ax.scatter(z_camino_2d[:, 0], z_camino_2d[:, 1], c=color, s=50, alpha=0.9)

        ax.scatter(
            z_inicio_2d[0],
            z_inicio_2d[1],
            c=color,
            s=250,
            marker="o",
            edgecolors="black",
            linewidth=2,
        )
        ax.scatter(
            z_fin_2d[0],
            z_fin_2d[1],
            c=color,
            s=250,
            marker="s",
            edgecolors="black",
            linewidth=2,
        )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Dimensión Latente 1", fontsize=12)
    ax.set_ylabel("Dimensión Latente 2", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    save_plot(fig, "results/multiple_interpolation_paths_clean.png")
    plt.show()


def plot_latent_space_emojis(latent_representations, emoji_chars, generated_samples=None, title="Latent Space Emoji Distribution"):
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (x, y) in enumerate(latent_representations):
        ax.scatter(x, y, s=100)
        ax.annotate(emoji_chars[i], (x, y), xytext=(5, 5), textcoords="offset points")

    if generated_samples is not None:
        ax.scatter(generated_samples[:, 0], generated_samples[:, 1], c="red", s=100, marker="x", label="Generados")

    ax.set_title(title)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.grid(True)
    if generated_samples is not None:
        ax.legend()
    fig.tight_layout()
    save_plot(fig, "results/latent_space_emojis.png")
    plt.show()


def plot_generated_emojis_for_paths(z_original, vae, image_size, emoji_chars, title="Generación de Emojis para Diferentes Caminos"):
    caminos = [
        (0, -1, "red", f"{emoji_chars[0]} → {emoji_chars[-1]}"),
        (1, -2, "green", f"{emoji_chars[1]} → {emoji_chars[-2]}"),
        (2, -3, "blue", f"{emoji_chars[2]} → {emoji_chars[-3]}"),
        (3, -4, "orange", f"{emoji_chars[3]} → {emoji_chars[-4]}"),
        (4, -5, "purple", f"{emoji_chars[4]} → {emoji_chars[-5]}"),
    ]

    num_steps = 5
    alphas = np.linspace(0, 1, num_steps)

    fig, axs = plt.subplots(len(caminos), num_steps, figsize=(15, 3 * len(caminos)))
    fig.suptitle(title, fontsize=16)

    for idx, (inicio_idx, fin_idx, color, label) in enumerate(caminos):
        z_inicio = z_original[inicio_idx]
        z_fin = z_original[fin_idx]

        z_camino = np.array([z_inicio * (1 - alpha) + z_fin * alpha for alpha in alphas])

        decoder_activations = vae.decoder.forward(z_camino)
        generated_camino = decoder_activations[-1]

        for i in range(num_steps):
            generated_img = generated_camino[i].reshape(image_size, image_size)
            axs[idx, i].imshow(generated_img, cmap="gray", vmin=0, vmax=1)
            axs[idx, i].set_title(f"{label}\nα = {alphas[i]:.2f}")
            axs[idx, i].axis("off")

        axs[idx, 0].set_ylabel(label, fontsize=12, rotation=0, ha="right", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "results/generated_emojis_for_paths.png")
    plt.show()


# TODO: si no se usa volarlo
def plot_latent_distribution_analysis(z, title="Análisis de la Distribución del Espacio Latente"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    axs[0, 0].hist(z.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black")
    axs[0, 0].axvline(x=-1, color="red", linestyle="--", alpha=0.7, label="Límite -1")
    axs[0, 0].axvline(x=1, color="red", linestyle="--", alpha=0.7, label="Límite +1")
    axs[0, 0].axvline(x=-1.96, color="orange", linestyle="--", alpha=0.7, label="95% N(0,1)")
    axs[0, 0].axvline(x=1.96, color="orange", linestyle="--", alpha=0.7, label="95% N(0,1)")
    axs[0, 0].set_title("Distribución de todos los valores latentes")
    axs[0, 0].set_xlabel("Valor latente")
    axs[0, 0].set_ylabel("Frecuencia")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].boxplot([z[:, i] for i in range(min(10, z.shape[1]))], labels=[f"Dim {i + 1}" for i in range(min(10, z.shape[1]))])
    axs[0, 1].axhline(y=-1, color="red", linestyle="--", alpha=0.7, label="Límite -1")
    axs[0, 1].axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Límite +1")
    axs[0, 1].set_title("Box plot por dimensión (primeras 10)")
    axs[0, 1].set_ylabel("Valor latente")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Estadísticas por dimensión
    means = np.mean(z, axis=0)
    stds = np.std(z, axis=0)
    mins = np.min(z, axis=0)
    maxs = np.max(z, axis=0)

    x_pos = np.arange(min(10, z.shape[1]))
    width = 0.35

    axs[1, 0].bar(x_pos - width / 2, means[:10], width, label="Media", alpha=0.7)
    axs[1, 0].bar(x_pos + width / 2, stds[:10], width, label="Desv. Est.", alpha=0.7)
    axs[1, 0].set_title("Media y Desviación Estándar por Dimensión")
    axs[1, 0].set_xlabel("Dimensión")
    axs[1, 0].set_ylabel("Valor")
    axs[1, 0].set_xticks(x_pos)
    axs[1, 0].set_xticklabels([f"Dim {i + 1}" for i in range(10)])
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Rango por dimensión
    ranges = maxs - mins
    axs[1, 1].bar(x_pos, ranges[:10], alpha=0.7, color="green")
    axs[1, 1].axhline(y=2, color="red", linestyle="--", alpha=0.7, label="Rango [-1,1]")
    axs[1, 1].set_title("Rango por Dimensión")
    axs[1, 1].set_xlabel("Dimensión")
    axs[1, 1].set_ylabel("Rango (max - min)")
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels([f"Dim {i + 1}" for i in range(10)])
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, weight="bold")
    plt.tight_layout()
    save_plot(fig, "results/latent_distribution_analysis.png")
    plt.show()
