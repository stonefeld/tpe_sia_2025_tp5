from ej1.src.activators import get_activation_function
from ej1.src.optimizers import get_optimizer
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
        "max_pixel_error": config.get("max_pixel_error", 1),
    }

    return init_params, train_params
