import matplotlib.pyplot as plt
import numpy as np

from shared.utils import save_plot


def plot_generated_emojis(generated_images, image_size):
    n_images = len(generated_images)
    rows = 2
    cols = 5

    fig, axs = plt.subplots(rows, cols, figsize=(10, 4))
    if n_images <= 1:
        axs = [axs]
    axs = axs.flatten()

    for i in range(n_images):
        axs[i].imshow(generated_images[i].reshape(image_size, image_size), cmap="gray")
        axs[i].axis("off")

    # Hide any unused subplots
    for i in range(n_images, len(axs)):
        axs[i].axis("off")

    fig.suptitle("Emojis generados")
    plt.tight_layout()
    save_plot(fig, "results/generated_emojis.png")
    plt.show()
