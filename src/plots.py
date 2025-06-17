import matplotlib.pyplot as plt


def plot_letter(data, index):
    letter = data[index].reshape(7, 5)
    plt.imshow(letter, cmap="binary")
    plt.axis("off")
    plt.show()


def plot_all_letters(data):
    n_letters = len(data)
    n_cols = 8
    n_rows = (n_letters + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 2 * n_rows))
    for i in range(n_letters):
        plt.subplot(n_rows, n_cols, i + 1)
        letter = data[i].reshape(7, 5)
        plt.imshow(letter, cmap="binary")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_latent_space(latent_representations, font_data):
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(latent_representations):
        if i == 31:  # DEL character
            char = "DEL"
        else:
            char = chr(0x60 + i)
        plt.scatter(x, y, s=100)
        plt.annotate(char, (x, y), xytext=(5, 5), textcoords="offset points")

    plt.title("Distribución de caracteres en el espacio latente")
    plt.xlabel("Dimensionalidad Latente 1")
    plt.ylabel("Dimensionalidad Latente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
