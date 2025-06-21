import os

import numpy as np


def pixel_error(original, reconstructed, threshold=0.5):
    reconstructed_bin = (reconstructed > threshold).astype(int)
    return np.sum(np.abs(original - reconstructed_bin), axis=1)


def save_plot(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filepath}")
