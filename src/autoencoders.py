import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def pixel_error(original, reconstructed, threshold=0.5):
    reconstructed_bin = (reconstructed > threshold).astype(int)
    return np.sum(np.abs(original - reconstructed_bin), axis=1)  # array of errors per sample


class Autoencoder:
    def __init__(self, input_dim=35, hidden_dim=8, latent_dim=2, learning_rate=0.01):
        # Initialize weights with Xavier/Glorot initialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Encoder weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, latent_dim))

        # Decoder weights
        self.W3 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b4 = np.zeros((1, input_dim))

    def forward(self, X):
        # Encoder
        self.hidden1 = sigmoid(np.dot(X, self.W1) + self.b1)
        self.latent = sigmoid(np.dot(self.hidden1, self.W2) + self.b2)

        # Decoder
        self.hidden2 = sigmoid(np.dot(self.latent, self.W3) + self.b3)
        self.output = sigmoid(np.dot(self.hidden2, self.W4) + self.b4)

        return self.output, self.latent

    def backward(self, X, output):
        # Calculate error
        error = X - output

        # Decoder gradients
        d_output = error * sigmoid_derivative(output)
        d_hidden2 = np.dot(d_output, self.W4.T) * sigmoid_derivative(self.hidden2)
        d_latent = np.dot(d_hidden2, self.W3.T) * sigmoid_derivative(self.latent)

        # Encoder gradients
        d_hidden1 = np.dot(d_latent, self.W2.T) * sigmoid_derivative(self.hidden1)

        # Update weights
        self.W4 += self.learning_rate * np.dot(self.hidden2.T, d_output)
        self.b4 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.W3 += self.learning_rate * np.dot(self.latent.T, d_hidden2)
        self.b3 += self.learning_rate * np.sum(d_hidden2, axis=0, keepdims=True)
        self.W2 += self.learning_rate * np.dot(self.hidden1.T, d_latent)
        self.b2 += self.learning_rate * np.sum(d_latent, axis=0, keepdims=True)
        self.W1 += self.learning_rate * np.dot(X.T, d_hidden1)
        self.b1 += self.learning_rate * np.sum(d_hidden1, axis=0, keepdims=True)

    def train(self, X, epochs=1000, batch_size=32, max_pixel_error=None):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                left = i
                right = left + batch_size
                batch_indices = indices[left:right]
                batch_X = X[batch_indices]
                output, _ = self.forward(batch_X)
                loss = np.mean(np.square(batch_X - output))
                total_loss += loss
                self.backward(batch_X, output)

            # Early stopping based on pixel error
            if max_pixel_error is not None:
                reconstructed, _ = self.forward(X)
                errors = pixel_error(X, reconstructed)
                max_err = np.max(errors)
                if (epoch + 1) % 100 == 0 or max_err <= max_pixel_error:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / (n_samples / batch_size):.10f}, Max Pixel Error: {max_err}")

                if max_err <= max_pixel_error:
                    print(f"Training stopped: max pixel error {max_err} <= {max_pixel_error}")
                    break

            elif (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / (n_samples / batch_size):.10f}")

    def get_latent_representations(self, X):
        _, latent = self.forward(X)
        return latent
