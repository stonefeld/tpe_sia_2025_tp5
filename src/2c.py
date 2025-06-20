# Adaptación de 2c.py para que funcione con la estructura actual del usuario
# usando la clase VariationalAutoEncoder que ya tiene definida

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from autoencoders import Autoencoder
from variationalAutoEncoder import VariationalAutoEncoder
from optimizers import Adam
from activators import sigmoid, sigmoid_prime

def render_emojis(size=32):
    emojis = {
        "grinning": "😀", "crying": "😢", "angry": "😠",
        "sunglasses": "😎", "neutral": "😐", "sleeping": "😴"
    }
    font_paths = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "C:/Windows/Fonts/seguiemj.ttf"
    ]
    emoji_font = None
    for path in font_paths:
        if os.path.exists(path):
            for s in [16, 20, 24, 28, 32]:
                try:
                    emoji_font = ImageFont.truetype(path, s)
                    break
                except OSError:
                    continue
            if emoji_font: break
    if emoji_font is None:
        print("⚠️ No se encontró fuente de emoji. Usando default.")
        emoji_font = ImageFont.load_default()

    data, names = [], []
    for name, char in emojis.items():
        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((2, 0), char, font=emoji_font, fill=0)
        arr = np.array(img).astype(np.float32).reshape(-1) / 255.0
        data.append(arr)
        names.append(name)
    return np.array(data), names

# === 1. Cargar datos ===
X, names = render_emojis()
input_size = 1024
hidden_size = 128
latent_size = 2

# Mostrar los emojis renderizados
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
axs = axs.flatten()  # ✅ Asegura que axs sea una lista plana
for i, arr in enumerate(X):
    axs[i].imshow(arr.reshape(32, 32), cmap='gray')
    axs[i].set_title(names[i], fontsize=12)
    axs[i].axis('off')
plt.suptitle("Emojis Unicode Renderizados (32x32)", fontsize=16)
plt.tight_layout()
plt.show()

# === 2. Crear encoder y decoder usando Autoencoder ===

# Encoder: input_size -> hidden_size -> latent_size*2 (mu y logvar)
encoder_layers = [input_size, hidden_size, latent_size * 2]
encoder = Autoencoder(encoder_layers, sigmoid, sigmoid_prime, Adam(learning_rate=0.001))

# Decoder: latent_size -> hidden_size -> input_size
decoder_layers = [latent_size, hidden_size, input_size]
decoder = Autoencoder(decoder_layers, sigmoid, sigmoid_prime, Adam(learning_rate=0.001))

# === 3. Crear VAE ===
vae = VariationalAutoEncoder(encoder, decoder, Adam(learning_rate=0.001), latent_size)

# === 4. Entrenamiento ===
X_train = X.astype(np.float32)
epochs = 1000
for epoch in range(epochs):
    loss = vae.train_batch(X_train)
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# === 5. Generación desde espacio latente ===
z_samples = np.random.normal(0, 1, size=(10, latent_size))
decoder_activations = decoder.forward(z_samples)
generated = decoder_activations[-1]  # Tomar la última activación

# === 6. Mostrar resultados ===
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated[i].reshape(32, 32), cmap='gray')
    plt.axis('off')
plt.suptitle("Emojis generados por el VAE (2.c)")
plt.tight_layout()
plt.savefig("vae_emojis_2c.png")
plt.show()
