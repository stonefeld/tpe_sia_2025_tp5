from typing import Tuple
import numpy as np

class VariationalAutoEncoder:
    def __init__(self, encoder, decoder, optimizer, latent_dim: int):
        self.encoder = encoder  # red para obtener mu y log_var
        self.decoder = decoder  # red para decodificar desde z
        self.optimizer = optimizer
        self.latent_dim = latent_dim

    def sample_latent(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*std.shape)
        return mu + eps * std

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Usar el método forward de Autoencoder que devuelve activaciones
        encoded_activations = self.encoder.forward(x)
        encoded = encoded_activations[-1]  # Última capa del encoder
        
        mu = encoded[:, :self.latent_dim]
        log_var = encoded[:, self.latent_dim:]
        z = self.sample_latent(mu, log_var)
        
        # Usar el método forward de Autoencoder para el decoder
        decoded_activations = self.decoder.forward(z)
        x_hat = decoded_activations[-1]  # Última capa del decoder
        
        return x_hat, mu, log_var, z

    def binary_cross_entropy(self, x: np.ndarray, x_hat: np.ndarray) -> float:
        # Evitar log(0) con +1e-8
        return -np.mean(x * np.log(x_hat + 1e-8) + (1 - x) * np.log(1 - x_hat + 1e-8))

    def kl_divergence(self, mu: np.ndarray, log_var: np.ndarray) -> float:
        return -0.5 * np.mean(1 + log_var - mu ** 2 - np.exp(log_var))

    def loss_fn(self, x: np.ndarray, x_hat: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> float:
        bce = self.binary_cross_entropy(x, x_hat)
        kl = self.kl_divergence(mu, log_var)
        return bce + kl

    def train_batch(self, x: np.ndarray) -> float:
        # Forward
        x_hat, mu, log_var, z = self.forward(x)

        # Loss
        loss = self.loss_fn(x, x_hat, mu, log_var)

        # Backprop desde decoder
        grad_xhat = -(x / (x_hat + 1e-8)) + ((1 - x) / (1 - x_hat + 1e-8))
        
        # Usar el método backward de Autoencoder para el decoder
        self.decoder.backward(x, self.decoder.forward(z))

        # Backprop hacia encoder
        # Para simplificar, usamos el mismo gradiente para mu y logvar
        grad_mu = grad_xhat[:, :self.latent_dim]
        grad_log_var = grad_xhat[:, self.latent_dim:]
        
        # Usar el método backward de Autoencoder para el encoder
        # self.encoder.backward(x, self.encoder.forward(x))

        return loss
