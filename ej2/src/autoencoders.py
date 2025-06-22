import copy
from typing import Tuple

import numpy as np

from shared.optimizers import SGD


class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, hidden_layers, learning_rate, tita, tita_prime, optimizer=SGD()):
        self.latent_dim = latent_dim
        self.encoder = Autoencoder(
            layers=[input_dim] + hidden_layers + [latent_dim * 2],
            tita=tita,
            tita_prime=tita_prime,
            optimizer=optimizer,
        )
        self.decoder = Autoencoder(
            layers=[latent_dim] + hidden_layers[::-1] + [input_dim],
            tita=tita,
            tita_prime=tita_prime,
            optimizer=copy.deepcopy(optimizer),
        )

    def reparametrize(self, mu, log_var):
        std = np.exp(0.5 * log_var)
        eps = np.random.normal(size=std.shape)
        return mu + std * eps

    def forward(self, x):
        encoder_activations = self.encoder.forward(x)
        encoded = encoder_activations[-1]

        mu = encoded[:, : self.latent_dim]
        log_var = encoded[:, self.latent_dim :]
        z = self.reparametrize(mu, log_var)

        decoder_activations = self.decoder.forward(z)
        reconstructed = decoder_activations[-1]

        return reconstructed, mu, log_var, z

    def binary_cross_entropy(self, x, x_hat):
        return -np.mean(x * np.log(x_hat + 1e-8) + (1 - x) * np.log(1 - x_hat + 1e-8))

    def kl_divergence(self, mu, log_var):
        return -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))

    def loss_fn(self, x, x_hat, mu, log_var):
        bce = self.binary_cross_entropy(x, x_hat)
        kl = self.kl_divergence(mu, log_var)
        return bce + kl

    def train(self, x, epochs=1000, batch_size=None):
        x = np.array([sample.flatten() for sample in x])
        n_samples = x.shape[0]
        batch_size = batch_size or n_samples

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                left = i
                right = min(left + batch_size, n_samples)
                batch_idx = idx[left:right]
                batch_x = x[batch_idx]

                reconstructed, mu, log_var, z = self.forward(batch_x)
                loss = self.loss_fn(batch_x, reconstructed, mu, log_var)
                total_loss += loss

                decoder_activations = self.decoder.forward(z)
                self.decoder.backward(batch_x, decoder_activations)

                encoder_activations = self.encoder.forward(batch_x)
                kl_grad_mu = mu
                kl_grad_log_var = 0.5 * (np.exp(log_var) - 1)
                encoder_target = np.concatenate([kl_grad_mu, kl_grad_log_var], axis=1)
                self.encoder.backward(encoder_target, encoder_activations)

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / (n_samples / batch_size)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")


class Autoencoder:
    def __init__(self, layers, tita, tita_prime, optimizer=SGD()):
        self.layers = layers
        self.tita = tita
        self.tita_prime = tita_prime
        self.optimizer = optimizer
        self.weights = []
        self.biases = []

        # Inicialización Xavier/Glorot
        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1  # +1 for bias
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)))

        self.optimizer.initialize(self.weights)

    def forward(self, x):
        input = np.array(x)
        activations = [input]

        for weight in self.weights:
            bias = np.ones((input.shape[0], 1))
            input_with_bias = np.concatenate((bias, input), axis=1)
            h = np.dot(input_with_bias, weight.T)
            output = np.array([self.tita(h_i) for h_i in h])
            activations.append(output)
            input = output

        return activations

    def backward(self, x, activations):
        deltas = [None] * len(self.weights)
        output = activations[-1]

        error = np.array(x) - output

        # Delta de la capa de salida
        deltas[-1] = error * np.array([self.tita_prime(o) for o in output])

        # Delta de las capas ocultas
        for i in reversed(range(len(deltas) - 1)):
            j = i + 1
            layer_output = activations[j]
            next_delta = deltas[j]
            next_weights = self.weights[j]

            # Calcular delta evitando bias
            hidden_error = np.dot(next_delta, next_weights[:, 1:])
            deltas[i] = hidden_error * np.array([self.tita_prime(o) for o in layer_output])

        # Actualizar pesos
        for i in range(len(self.weights)):
            batch_size = x.shape[0]
            weight_gradients = np.zeros_like(self.weights[i])

            for b in range(batch_size):
                layer_input = np.concatenate(([1], activations[i][b]))
                weight_gradients += np.outer(deltas[i][b], layer_input)

            weight_gradients /= batch_size
            self.optimizer.update(i, self.weights[i], weight_gradients)

    def train(self, x, epochs=1000, batch_size=None):
        x = np.array([sample.flatten() for sample in x])
        n_samples = x.shape[0]
        batch_size = batch_size or n_samples

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                left = i
                right = left + batch_size
                batch_idx = idx[left:right]
                batch_x = x[batch_idx]
                activations = self.forward(batch_x)
                self.backward(batch_x, activations)

                total_loss += np.mean(np.square(batch_x - activations[-1]))

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / (n_samples / batch_size):.10f}")
