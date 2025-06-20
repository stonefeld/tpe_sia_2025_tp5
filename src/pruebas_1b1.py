import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os

# Agregar el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autoencoders import Autoencoder
from src.activators import sigmoid, sigmoid_prime
from src.optimizers import Adam
from src.utils import pixel_error

# ----------- 1. Cargar datos de font -----------

# Usar los mismos datos que están en main.py
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

# ----------- 2. Definir arquitecturas a testear -----------

architectures = [
    {"layers": [35, 16, 2, 16, 35]},
    {"layers": [35, 32, 2, 32, 35]},
    {"layers": [35, 32, 16, 2, 16, 32, 35]},
    {"layers": [35, 64, 32, 4, 32, 64, 35]},
    {"layers": [35, 64, 32, 8, 32, 64, 35]}
]

# ----------- 3. Métricas por arquitectura -----------

results = {"arch": [], "MSE": [], "Pixel_Error": [], "Max_Error": []}

# Cargar datos
images = decode_font(font_data)
flattened = images  # Ya está aplanado

for arch in architectures:
    print(f"\nEntrenando arquitectura: {arch['layers']}")
    
    # Crear y entrenar autoencoder
    optimizer = Adam(learning_rate=0.001, layers=arch['layers'])
    model = Autoencoder(
        layers=arch['layers'], 
        tita=sigmoid, 
        tita_prime=sigmoid_prime, 
        optimizer=optimizer
    )
    
    # Entrenamiento
    model.train(flattened, epochs=1000, batch_size=8, max_pixel_error=None)
    
    # Evaluación
    reconstructed = model.forward(flattened)[-1]
    errors = pixel_error(flattened, reconstructed)
    
    # Calcular métricas
    mse = np.mean(np.square(flattened - reconstructed))
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    results["arch"].append(f"{arch['layers']}")
    results["MSE"].append(mse)
    results["Pixel_Error"].append(mean_error)
    results["Max_Error"].append(max_error)
    
    print(f"MSE: {mse:.6f}, Mean Pixel Error: {mean_error:.2f}, Max Pixel Error: {max_error}")

# ----------- 4. Graficar resultados -----------

plt.figure(figsize=(15, 10))

for i, metric in enumerate(["MSE", "Pixel_Error", "Max_Error"]):
    plt.subplot(2, 2, i + 1)
    plt.bar(range(len(results[metric])), results[metric])
    plt.title(metric)
    plt.xlabel("Arquitectura")
    plt.ylabel(metric)
    plt.xticks(range(len(results["arch"])), [f"Arch {i+1}" for i in range(len(results["arch"]))], rotation=45)

# Mostrar arquitecturas
plt.subplot(2, 2, 4)
plt.text(0.1, 0.9, "Arquitecturas:", fontsize=12, fontweight='bold')
for i, arch in enumerate(results["arch"]):
    plt.text(0.1, 0.8 - i*0.1, f"Arch {i+1}: {arch}", fontsize=10)
plt.axis('off')

plt.suptitle("Comparación de Arquitecturas de Autoencoder", fontsize=14)
plt.tight_layout()
plt.show()

# ----------- 5. Mostrar mejores resultados -----------

best_idx = np.argmin(results["MSE"])
print(f"\nMejor arquitectura: {results['arch'][best_idx]}")
print(f"MSE: {results['MSE'][best_idx]:.6f}")
print(f"Mean Pixel Error: {results['Pixel_Error'][best_idx]:.2f}")
print(f"Max Pixel Error: {results['Max_Error'][best_idx]}")
