import numpy as np
import sys
import os

# Agregar el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activators import sigmoid, sigmoid_prime
from src.optimizers import SGD
from src.utils import pixel_error


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
            bias = np.ones((input.shape[0], 1))  # bias for each sample
            input_with_bias = np.concatenate((bias, input), axis=1)  # bias
            h = np.dot(input_with_bias, weight.T)  # input_with_bias: (batch_size, input_size+1), weight.T: (input_size+1, output_size)
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
            # For batch processing, we need to handle each sample in the batch
            batch_size = x.shape[0]
            weight_gradients = np.zeros_like(self.weights[i])

            for b in range(batch_size):
                layer_input = np.concatenate(([1], activations[i][b]))
                weight_gradients += np.outer(deltas[i][b], layer_input)

            # Average gradients over batch
            weight_gradients /= batch_size
            self.optimizer.update(i, self.weights[i], weight_gradients)

    def train(self, x, epochs=1000, batch_size=32, max_pixel_error=None):
        x = np.array([sample.flatten() for sample in x])
        n_samples = x.shape[0]

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

            # Corte temprano basado en error de píxeles
            if max_pixel_error is not None:
                activations = self.forward(x)
                errors = pixel_error(x, activations[-1])
                max_error = np.max(errors)

                if (epoch + 1) % 100 == 0 or max_error <= max_pixel_error:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / (n_samples / batch_size):.10f}, Max Pixel Error: {max_error}")

                if max_error <= max_pixel_error:
                    print(f"Training stopped: max pixel error {max_error} <= {max_pixel_error}")
                    break

            elif (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / (n_samples / batch_size):.10f}")

    def get_latent_representations(self, x):
        x = np.array([sample.flatten() for sample in x])
        activations = self.forward(x)
        return activations[len(self.layers) // 2]
