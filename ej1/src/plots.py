from collections import Counter

import matplotlib.pyplot as plt

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
