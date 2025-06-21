import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Agregar el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activators import sigmoid, sigmoid_prime
from src.autoencoders import Autoencoder
from src.optimizers import Adam

# ----------- 1. Cargar datos de font (igual que en pruebas_1b1.py) -----------

font_data = [
    [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],  # 0x60, `
    [0x00, 0x0E, 0x01, 0x0D, 0x13, 0x13, 0x0D],  # 0x61, a
    [0x10, 0x10, 0x10, 0x1C, 0x12, 0x12, 0x1C],  # 0x62, b
    [0x00, 0x00, 0x00, 0x0E, 0x10, 0x10, 0x0E],  # 0x63, c
    [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],  # 0x64, d
    [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0F],  # 0x65, e
    [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],  # 0x66, f
    [0x0E, 0x11, 0x13, 0x0D, 0x01, 0x01, 0x0E],  # 0x67, g
    [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],  # 0x68, h
    [0x00, 0x04, 0x00, 0x0C, 0x04, 0x04, 0x0E],  # 0x69, i
    [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],  # 0x6a, j
    [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],  # 0x6b, k
    [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],  # 0x6c, l
    [0x00, 0x00, 0x0A, 0x15, 0x15, 0x11, 0x11],  # 0x6d, m
    [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],  # 0x6e, n
    [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],  # 0x6f, o
    [0x00, 0x1C, 0x12, 0x12, 0x1C, 0x10, 0x10],  # 0x70, p
    [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],  # 0x71, q
    [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],  # 0x72, r
    [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],  # 0x73, s
    [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],  # 0x74, t
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],  # 0x75, u
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],  # 0x76, v
    [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],  # 0x77, w
    [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],  # 0x78, x
    [0x00, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E],  # 0x79, y
    [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],  # 0x7a, z
    [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],  # 0x7b, {
    [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],  # 0x7c, |
    [0x0C, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0C],  # 0x7d, }
    [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],  # 0x7e, ~
    [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F],  # 0x7f, DEL
]


def decode_font(font_data):
    images = []
    for glyph in font_data:
        rows = []
        for value in glyph:
            bin_str = format(value, "05b")
            rows.extend([int(b) for b in bin_str])

        images.append(rows)

    return np.array(images, dtype=np.float32)


# ----------- 2. Crear y entrenar modelo -----------

print("Cargando datos de font...")
images = decode_font(font_data)
flattened = images  # Ya está aplanado

print("Creando y entrenando autoencoder...")
# Usar una arquitectura simple con espacio latente de 2 dimensiones
layers = [35, 32, 16, 2, 16, 32, 35]
optimizer = Adam(learning_rate=0.001, layers=layers)
model = Autoencoder(layers=layers, tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)

# Aumentar épocas para un mejor entrenamiento
model.train(flattened, epochs=100000, batch_size=8, max_pixel_error=None)
print("Entrenamiento completado!")

# ----------- 3. Función para generar desde el espacio latente -----------


def generate_from_latent(model, z_values):
    """
    Genera imágenes desde puntos en el espacio latente usando el decoder del autoencoder
    """
    generated_images = []

    for z in z_values:
        # Usar el z como entrada directa para las capas del decoder
        current_input = z.reshape(1, -1)

        # Aplicar las capas del decoder
        decoder_start = len(model.layers) // 2
        for i in range(decoder_start, len(model.layers) - 1):
            bias = np.ones((current_input.shape[0], 1))
            input_with_bias = np.concatenate((bias, current_input), axis=1)
            h = np.dot(input_with_bias, model.weights[i].T)
            current_input = np.array([sigmoid(h_i) for h_i in h])

        generated_images.append(current_input[0])

    return np.array(generated_images)


# ----------- 4. Generar y mostrar resultados -----------

print("\nGenerando representaciones latentes de datos reales...")
latent_repr = model.get_latent_representations(flattened)

# ----------- 4.1. Interpolar en el espacio latente para ver transiciones suaves -----------

# Interpolar entre dos letras para ver una transición suave
z_start = latent_repr[1]  # letra 'a'
z_end = latent_repr[26]  # letra 'z'

print("\nInterpolando en el espacio latente entre 'a' y 'z'...")
alphas = np.linspace(0, 1, 5)
z_values = np.array([z_start * (1 - alpha) + z_end * alpha for alpha in alphas])

# Convertir a array de NumPy para facilitar el manejo
z_values = np.array(z_values)

print("\nGenerando imágenes desde el espacio latente interpolado...")
generated = generate_from_latent(model, z_values)

# Mostrar resultados: una única fila con la transición generada
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Imágenes Generadas por Interpolación en Espacio Latente (de 'a' a 'z')")

for i in range(5):
    generated_img = generated[i].reshape(7, 5)
    axs[i].imshow(generated_img, cmap="binary", vmin=0, vmax=1)
    axs[i].set_title(f"Paso de interpolación {i + 1}\nα = {alphas[i]:.2f}")
    axs[i].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ----------- 5. Mostrar el espacio latente completo con puntos de interpolación -----------

plt.figure(figsize=(8, 8))
# Dibujar todas las representaciones latentes de las letras reales
plt.scatter(latent_repr[:, 0], latent_repr[:, 1], alpha=0.6, label="Letras Reales")

# Resaltar los puntos de interpolación usados
plt.scatter(z_values[:, 0], z_values[:, 1], c="red", s=100, marker="x", label="Puntos Interpolados")

# Resaltar inicio y fin
plt.scatter(z_start[0], z_start[1], c="green", s=150, marker="o", edgecolors="k", label="Inicio ('a')")
plt.scatter(z_end[0], z_end[1], c="purple", s=150, marker="o", edgecolors="k", label="Fin ('z')")

plt.title("Espacio Latente con Puntos de Interpolación")
plt.xlabel("Dimensión Latente 1")
plt.ylabel("Dimensión Latente 2")
plt.legend()
plt.grid(True)
plt.show()

# ----------- 6. Visualización completa del espacio latente con múltiples caminos -----------

print("\nCreando visualización completa del espacio latente...")

# Crear una figura más grande para mostrar múltiples caminos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Gráfico 1: Espacio latente con todas las letras etiquetadas
ax1.scatter(latent_repr[:, 0], latent_repr[:, 1], c="blue", s=100, alpha=0.7)
ax1.set_title("Espacio Latente Completo - Todas las Letras", fontsize=14)
ax1.set_xlabel("Dimensión Latente 1")
ax1.set_ylabel("Dimensión Latente 2")
ax1.grid(True, alpha=0.3)

# Etiquetar algunas letras importantes
letras_importantes = [1, 5, 15, 26]  # a, e, o, z
for i in letras_importantes:
    ax1.annotate(
        chr(0x61 + i - 1), (latent_repr[i, 0], latent_repr[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=12, fontweight="bold"
    )

# Gráfico 2: Múltiples caminos de interpolación
ax2.scatter(latent_repr[:, 0], latent_repr[:, 1], c="lightblue", s=80, alpha=0.6, label="Letras Reales")

# Definir varios caminos de interpolación
caminos = [
    (1, 26, "red", "a → z"),  # a a z
    (5, 15, "green", "e → o"),  # e a o
    (1, 15, "orange", "a → o"),  # a a o
    (5, 26, "purple", "e → z"),  # e a z
]

for inicio, fin, color, label in caminos:
    z_inicio = latent_repr[inicio]
    z_fin = latent_repr[fin]

    # Crear puntos de interpolación
    alphas_camino = np.linspace(0, 1, 10)
    z_camino = np.array([z_inicio * (1 - alpha) + z_fin * alpha for alpha in alphas_camino])

    # Dibujar el camino
    ax2.plot(z_camino[:, 0], z_camino[:, 1], c=color, linewidth=2, alpha=0.7, label=label)
    ax2.scatter(z_camino[:, 0], z_camino[:, 1], c=color, s=30, alpha=0.8)

    # Marcar inicio y fin del camino
    ax2.scatter(z_inicio[0], z_inicio[1], c=color, s=150, marker="o", edgecolors="black", linewidth=2)
    ax2.scatter(z_fin[0], z_fin[1], c=color, s=150, marker="s", edgecolors="black", linewidth=2)

ax2.set_title("Múltiples Caminos de Interpolación en el Espacio Latente", fontsize=14)
ax2.set_xlabel("Dimensión Latente 1")
ax2.set_ylabel("Dimensión Latente 2")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------- 7. Generar imágenes para todos los caminos -----------

print("\nGenerando imágenes para todos los caminos de interpolación...")

fig, axs = plt.subplots(len(caminos), 5, figsize=(15, 3 * len(caminos)))
fig.suptitle("Generación de Imágenes para Diferentes Caminos en el Espacio Latente", fontsize=16)

for idx, (inicio, fin, color, label) in enumerate(caminos):
    z_inicio = latent_repr[inicio]
    z_fin = latent_repr[fin]

    # Crear puntos de interpolación
    alphas_camino = np.linspace(0, 1, 5)
    z_camino = np.array([z_inicio * (1 - alpha) + z_fin * alpha for alpha in alphas_camino])

    # Generar imágenes
    generated_camino = generate_from_latent(model, z_camino)

    # Mostrar imágenes
    for i in range(5):
        generated_img = generated_camino[i].reshape(7, 5)
        axs[idx, i].imshow(generated_img, cmap="binary", vmin=0, vmax=1)
        axs[idx, i].set_title(f"{label}\nα = {alphas_camino[i]:.2f}")
        axs[idx, i].axis("off")

    # Agregar etiqueta de fila
    axs[idx, 0].set_ylabel(label, fontsize=12, rotation=0, ha="right", va="center")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
