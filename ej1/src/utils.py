import numpy as np
from sklearn.decomposition import PCA

from shared.activators import get_activation_function
from shared.metrics import compute_mse
from shared.optimizers import get_optimizer
from shared.utils import load_config


def build_params(config_file):
    config = load_config(config_file)

    tita, tita_prime = get_activation_function(config.get("activation_function", "sigmoid"))

    init_params = {
        "layers": config.get("layers", [35, 10, 2, 10, 35]),
        "tita": tita,
        "tita_prime": tita_prime,
        "optimizer": get_optimizer(config.get("optimizer", "sgd"), **config.get("optimizer_opts", {})),
    }

    train_params = {
        "epochs": config.get("epochs", 100000),
        "batch_size": config.get("batch_size", 8),
        "max_pixel_error": config.get("max_pixel_error", None),
    }

    return init_params, train_params


def pca_2d(latent_representations):
    pca = PCA(n_components=2)
    return pca.fit_transform(latent_representations)


def generate_from_latent(model, z_values):
    generated_images = []

    for z in z_values:
        current_input = z.reshape(1, -1)
        decoder_start = len(model.layers) // 2

        for i in range(decoder_start, len(model.layers) - 1):
            bias = np.ones((current_input.shape[0], 1))
            input_with_bias = np.concatenate((bias, current_input), axis=1)
            h = np.dot(input_with_bias, model.weights[i].T)
            current_input = np.array([model.tita(h_i) for h_i in h])

        generated_images.append(current_input[0])

    return np.array(generated_images)


def compute_all_metrics(original, reconstructed):
    metrics = {
        "MSE": compute_mse(original, reconstructed),
    }

    return metrics


def print_metrics_summary(results):
    best_idx = np.argmin(results["MSE"])

    print(f"\nMejor arquitectura: {results['arch'][best_idx]}")

    for metric in results.keys():
        if metric != "arch":
            value = results[metric][best_idx]
            if isinstance(value, float):
                if metric in ["MSE", "SSIM"]:
                    print(f"{metric}: {value:.6f}")
                elif metric == "PSNR":
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")

    return best_idx
