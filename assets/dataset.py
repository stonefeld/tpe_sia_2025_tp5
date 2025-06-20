from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os

# Crear imágenes 32x32 renderizando emojis Unicode
emojis = {
    "grinning": "😀",
    "crying": "😢",
    "angry": "😠",
    "sunglasses": "😎",
    "neutral": "😐",
    "sleeping": "😴"
}

# Intentar usar una fuente con soporte de emojis
# Este bloque debería adaptarse a tu sistema si lo corrés localmente
font_paths = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # MacOS
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (Ubuntu)
    "C:/Windows/Fonts/seguiemj.ttf"  # Windows
]

emoji_font = None
for path in font_paths:
    if os.path.exists(path):
        try:
            # Intentar con diferentes tamaños si el primero falla
            for size in [16, 20, 24, 28, 32]:
                try:
                    emoji_font = ImageFont.truetype(path, size)
                    print(f"Fuente cargada exitosamente: {path} con tamaño {size}")
                    break
                except OSError as e:
                    if "invalid pixel size" in str(e):
                        continue
                    else:
                        raise e
            if emoji_font is not None:
                break
        except Exception as e:
            print(f"Error al cargar fuente {path}: {e}")
            continue

# Fallback si no hay fuente de emoji
if emoji_font is None:
    print("Usando fuente por defecto")
    emoji_font = ImageFont.load_default()

emoji_arrays = {}

# Renderizado en imágenes 32x32
for name, char in emojis.items():
    img = Image.new("L", (32, 32), color=255)  # Fondo blanco
    draw = ImageDraw.Draw(img)
    draw.text((2, 0), char, font=emoji_font, fill=0)  # Texto en negro
    arr = np.array(img) / 255.0  # Normalizar
    emoji_arrays[name] = arr

# Mostrar resultados
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
axs = axs.flatten()

for i, (name, arr) in enumerate(emoji_arrays.items()):
    axs[i].imshow(arr, cmap='gray')
    axs[i].set_title(f"{name}", fontsize=12)
    axs[i].axis('off')

plt.suptitle("Emojis Unicode Renderizados (32x32)", fontsize=16)
plt.tight_layout()
plt.show()
