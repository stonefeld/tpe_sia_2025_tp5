from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from shared.utils import save_plot


def plot_letter(data, index):
    fig, ax = plt.subplots()
    letter = data[index].reshape(7, 5)
    ax.imshow(letter, cmap="binary")
    ax.axis("off")
    fig.tight_layout()
    save_plot(fig, f"results/letter_{index}.png")
    plt.show()


def plot_all_letters(data):
    n_letters = len(data)
    n_cols = 8
    n_rows = (n_letters + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axs = axs.flatten()

    for i in range(n_letters):
        letter = data[i].reshape(7, 5)
        axs[i].imshow(letter, cmap="binary")
        axs[i].axis("off")

    fig.tight_layout()
    save_plot(fig, "results/letter_map.png")
    plt.show()


def plot_latent_space(latent_representations, font_data):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (x, y) in enumerate(latent_representations):
        if i == 31:  # DEL character
            char = "DEL"
        else:
            char = chr(0x60 + i)
        ax.scatter(x, y, s=100)
        ax.annotate(char, (x, y), xytext=(5, 5), textcoords="offset points")

    ax.set_title("Distribución de caracteres en el espacio latente")
    ax.set_xlabel("Dimensionalidad Latente 1")
    ax.set_ylabel("Dimensionalidad Latente 2")
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "results/latent_space.png")
    plt.show()


def plot_error_distribution(errors):
    error_counts = Counter(errors)
    xs = sorted(error_counts.keys())
    ys = [error_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(xs, ys)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(y), ha="center", va="bottom")

    ax.set_xlabel("Error (píxeles)")
    ax.set_ylabel("Cantidad de letras")
    ax.set_title("Distribución de errores por letra")
    fig.tight_layout()
    save_plot(fig, "results/error_distribution.png")
    plt.show()


def plot_interpolated_images(generated_images, alphas, title="Imágenes Generadas por Interpolación en Espacio Latente"):
    fig, axs = plt.subplots(1, len(generated_images), figsize=(15, 4))
    fig.suptitle(title)

    for i in range(len(generated_images)):
        generated_img = generated_images[i].reshape(7, 5)
        axs[i].imshow(generated_img, cmap="binary", vmin=0, vmax=1)
        axs[i].set_title(f"Paso de interpolación {i + 1}\nα = {alphas[i]:.2f}")
        axs[i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "results/interpolated_images.png")
    plt.show()


def plot_latent_space_with_interpolation(latent_repr, z_values, z_start, z_end):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(latent_repr[:, 0], latent_repr[:, 1], alpha=0.6, label="Letras Reales")
    ax.scatter(z_values[:, 0], z_values[:, 1], c="red", s=100, marker="x", label="Puntos Interpolados")

    ax.scatter(z_start[0], z_start[1], c="green", s=150, marker="o", edgecolors="k", label="Inicio ('a')")
    ax.scatter(z_end[0], z_end[1], c="purple", s=150, marker="o", edgecolors="k", label="Fin ('z')")

    ax.set_title("Espacio Latente con Puntos de Interpolación")
    ax.set_xlabel("Dimensión Latente 1")
    ax.set_ylabel("Dimensión Latente 2")
    ax.legend()
    ax.grid(True)

    save_plot(fig, "results/latent_space_with_interpolation.png")
    plt.show()


def plot_complete_latent_space_visualization(latent_repr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.scatter(latent_repr[:, 0], latent_repr[:, 1], c="blue", s=100, alpha=0.7)
    ax1.set_title("Espacio Latente Completo - Todas las Letras", fontsize=14)
    ax1.set_xlabel("Dimensión Latente 1")
    ax1.set_ylabel("Dimensión Latente 2")
    ax1.grid(True, alpha=0.3)

    letras_importantes = [1, 5, 15, 26]  # a, e, o, z
    for i in letras_importantes:
        ax1.annotate(
            chr(0x61 + i - 1),
            (latent_repr[i, 0], latent_repr[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    ax2.scatter(latent_repr[:, 0], latent_repr[:, 1], c="lightblue", s=80, alpha=0.6, label="Letras Reales")

    caminos = [
        (1, 26, "red", "a → z"),  # a to z
        (5, 15, "green", "e → o"),  # e to o
        (1, 15, "orange", "a → o"),  # a to o
        (5, 26, "purple", "e → z"),  # e to z
    ]

    for inicio, fin, color, label in caminos:
        z_inicio = latent_repr[inicio]
        z_fin = latent_repr[fin]

        alphas_camino = np.linspace(0, 1, 10)
        z_camino = np.array([z_inicio * (1 - alpha) + z_fin * alpha for alpha in alphas_camino])

        ax2.plot(z_camino[:, 0], z_camino[:, 1], c=color, linewidth=2, alpha=0.7, label=label)
        ax2.scatter(z_camino[:, 0], z_camino[:, 1], c=color, s=30, alpha=0.8)

        ax2.scatter(z_inicio[0], z_inicio[1], c=color, s=150, marker="o", edgecolors="black", linewidth=2)
        ax2.scatter(z_fin[0], z_fin[1], c=color, s=150, marker="s", edgecolors="black", linewidth=2)

    ax2.set_title("Múltiples Caminos de Interpolación en el Espacio Latente", fontsize=14)
    ax2.set_xlabel("Dimensión Latente 1")
    ax2.set_ylabel("Dimensión Latente 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "results/complete_latent_space_visualization.png")
    plt.show()


def plot_generated_images_for_paths(model, latent_repr, generate_from_latent_func):
    caminos = [
        (1, 26, "red", "a → z"),  # a to z
        (5, 15, "green", "e → o"),  # e to o
        (1, 15, "orange", "a → o"),  # a to o
        (5, 26, "purple", "e → z"),  # e to z
    ]

    fig, axs = plt.subplots(len(caminos), 5, figsize=(15, 3 * len(caminos)))
    fig.suptitle("Generación de Imágenes para Diferentes Caminos en el Espacio Latente", fontsize=16)

    for idx, (inicio, fin, color, label) in enumerate(caminos):
        z_inicio = latent_repr[inicio]
        z_fin = latent_repr[fin]

        alphas_camino = np.linspace(0, 1, 5)
        z_camino = np.array([z_inicio * (1 - alpha) + z_fin * alpha for alpha in alphas_camino])

        generated_camino = generate_from_latent_func(model, z_camino)

        for i in range(5):
            generated_img = generated_camino[i].reshape(7, 5)
            axs[idx, i].imshow(generated_img, cmap="binary", vmin=0, vmax=1)
            axs[idx, i].set_title(f"{label}\nα = {alphas_camino[i]:.2f}")
            axs[idx, i].axis("off")

        axs[idx, 0].set_ylabel(label, fontsize=12, rotation=0, ha="right", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "results/generated_images_for_paths.png")
    plt.show()


def plot_architecture_comparison(results, title="Comparación de Arquitecturas de Autoencoder"):
    """Plot comparison of different architectures with their metrics."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot each metric
    metrics = ["MSE", "Pixel_Error", "Max_Error"]
    for i, metric in enumerate(metrics):
        if metric in results:
            axs[0, i].bar(range(len(results[metric])), results[metric])
            axs[0, i].set_title(metric)
            axs[0, i].set_xlabel("Arquitectura")
            axs[0, i].set_ylabel(metric)
            axs[0, i].set_xticks(range(len(results["arch"])))
            axs[0, i].set_xticklabels([f"Arch {i+1}" for i in range(len(results["arch"]))], rotation=45)

    # Show architectures
    axs[0, 1].text(0.1, 0.9, "Arquitecturas:", fontsize=12, fontweight="bold")
    for i, arch in enumerate(results["arch"]):
        axs[0, 1].text(0.1, 0.8 - i * 0.1, f"Arch {i+1}: {arch}", fontsize=10)
    axs[0, 1].axis("off")

    # Plot additional metrics if available
    additional_metrics = ["PSNR", "SSIM", "Pixel_Accuracy"]
    metric_idx = 0
    for metric in additional_metrics:
        if metric in results and metric_idx < 2:
            row, col = 1, metric_idx
            axs[row, col].bar(range(len(results[metric])), results[metric])
            axs[row, col].set_title(metric)
            axs[row, col].set_xlabel("Arquitectura")
            axs[row, col].set_ylabel(metric)
            axs[row, col].set_xticks(range(len(results["arch"])))
            axs[row, col].set_xticklabels([f"Arch {i+1}" for i in range(len(results["arch"]))], rotation=45)
            metric_idx += 1

    # Hide unused subplot
    if metric_idx < 2:
        axs[1, 1].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    save_plot(fig, "results/architecture_comparison.png")
    plt.show()
