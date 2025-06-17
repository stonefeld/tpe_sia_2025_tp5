import matplotlib.pyplot as plt


def plot_letter(decoded_data, letter_index):
    letter = decoded_data[letter_index].reshape(7, 5)  # Reshape to 7x5 grid

    plt.figure(figsize=(3, 4))
    plt.imshow(letter, cmap="binary")
    plt.axis("off")
    plt.title(f"Letter at index {letter_index}")
    plt.show()


def plot_all_letters(decoded_data):
    n_letters = len(decoded_data)
    n_cols = 8
    n_rows = (n_letters + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 20))

    for i in range(n_letters):
        plt.subplot(n_rows, n_cols, i + 1)
        letter = decoded_data[i].reshape(7, 5)
        plt.imshow(letter, cmap="binary")
        plt.axis("off")
        plt.title(f"Index {i}")

    plt.tight_layout()
    plt.show()

