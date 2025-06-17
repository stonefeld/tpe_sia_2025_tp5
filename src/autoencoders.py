import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


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

    def train(self, X, epochs=1000, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            total_loss = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                left = i
                right = i + batch_size
                batch_indices = indices[left:right]
                batch_X = X[batch_indices]

                # Forward pass
                output, _ = self.forward(batch_X)

                # Calculate loss
                loss = np.mean(np.square(batch_X - output))
                total_loss += loss

                # Backward pass
                self.backward(batch_X, output)

            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / (n_samples / batch_size)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def get_latent_representations(self, X):
        _, latent = self.forward(X)
        return latent
