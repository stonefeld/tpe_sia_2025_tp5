<h1 align="center">Sistemas de Inteligencia Artificial</h1>
<h3 align="center">TP5: Autoencoders y VAE</h3>
<h4 align="center">Primer cuatrimestre 2025</h4>

# Requisitos

* Python ([versión 3.12.9](https://www.python.org/downloads/release/python-3129/))
* [UV](https://docs.astral.sh/uv/getting-started/installation/)

# Instalando las dependencias

```bash
# Si python 3.12.9 no esta instalado se puede instalar haciendo
uv python install 3.12.9

# Para crear y activar el entorno virtual
uv venv
source .venv/bin/activate  # En Unix
.venv\Scripts\activate     # En Windows

# Para instalar las dependencias
uv sync
```

# Corriendo el proyecto

El proyecto consta de diferentes ejercicios que implementan autoencoders clásicos y Variational Autoencoders (VAE). Cada ejercicio puede ejecutarse como un módulo de Python.

## Ejercicio 1a - Autoencoder Básico

Implementa un autoencoder básico para comprimir representaciones de letras a 2 dimensiones y reconstruirlas.

**Configuración:**
```
capas = [35, 64, 32, 2, 32, 64, 35]
epochs = 5000
batch_size = 8
learning_rate = 0.01
optimizer = adam
activator = sigmoid
```

**Ejecución:**
```bash
uv run -m ej1.ej1a configs/ej1_base.json
```

## Ejercicio 1b2 - Autoencoder con Ruido Salt & Pepper

Evalúa la robustez del autoencoder ante diferentes niveles de ruido salt & pepper (0.0 a 1.0).

**Configuración:**
```
capas = [35, 64, 32, 8, 32, 64, 35]
epochs = 3000
batch_size = 8
learning_rate = 0.01
optimizer = adam
activator = sigmoid
```

**Ejecución:**
```bash
uv run -m ej1.ej1b2
```

## Ejercicio 2c - Variational Autoencoder (VAE)

Implementa un VAE para generar y manipular emojis en el espacio latente, incluyendo:
- Generación de emojis desde distribución aprendida
- Interpolación en el espacio latente
- Visualización del espacio latente con PCA

**Configuración:**
```
capas = [784, 512, 256, 50, 256, 512, 784]
epochs = 3000
batch_size = 32
learning_rate = 0.0001
optimizer = adam
activator = sigmoid
```

**Ejecución:**
```bash
uv run -m ej2.ej2c
```

# Utilizando los motores de aprendizaje

Si se desea utilizar un _frontend_ propio pero utilizando el motor de este
proyecto, se puede hacer lo siguiente para cada caso:

```python
from ej1.src.autoencoders import Autoencoder
from ej2.src.autoencoders import VariationalAutoencoder
from shared.activators import sigmoid, sigmoid_prime
from shared.optimizers import Adam

# Para Autoencoder clásico
layers = [35, 64, 32, 2, 32, 64, 35]
optimizer = Adam(learning_rate=0.01, layers=layers)
autoencoder = Autoencoder(layers=layers, tita=sigmoid, tita_prime=sigmoid_prime, optimizer=optimizer)
autoencoder.train(data, epochs=5000, batch_size=8)
reconstructed = autoencoder.forward(data)[-1]

# Para VAE
vae = VariationalAutoencoder(
    input_dim=784,
    latent_dim=50,
    hidden_layers=[512, 256],
    tita=sigmoid,
    tita_prime=sigmoid_prime,
    optimizer=Adam(learning_rate=0.0001)
)
loss_history = vae.train(data, epochs=3000, batch_size=32)
_, _, z, _, _ = vae.forward(data)
```

# Estructura del Proyecto

```
tpe_sia_2025_tp5/
├── assets/                 # Datasets y recursos
│   ├── emoji_dataset.pkl   # Dataset de emojis
│   ├── font.h             # Datos de fuentes
│   └── fonts.ipynb        # Notebook de exploración
├── configs/               # Archivos de configuración
│   └── ej1_base.json      # Configuración base para ejercicio 1
├── ej1/                   # Ejercicio 1 - Autoencoders
│   ├── ej1a.py           # Autoencoder básico
│   ├── ej1a4.py          # Variante del ejercicio 1a
│   ├── ej1b1.py          # Ejercicio 1b1
│   ├── ej1b2.py          # Ejercicio 1b2 - Ruido salt & pepper
│   └── src/              # Código fuente del ejercicio 1
├── ej2/                   # Ejercicio 2 - VAE
│   ├── ej2c.py           # Variational Autoencoder
│   └── src/              # Código fuente del ejercicio 2
├── shared/                # Código compartido
│   ├── activators.py     # Funciones de activación
│   ├── metrics.py        # Métricas de evaluación
│   ├── optimizers.py     # Optimizadores
│   └── utils.py          # Utilidades generales
├── results/              # Resultados y gráficos
└── pyproject.toml        # Configuración del proyecto
```

# Características Principales

## Funciones de Activación
- **Sigmoid**: Función de activación principal utilizada en todos los ejercicios

## Optimizadores
- **Adam**: Optimizador utilizado con diferentes learning rates según el ejercicio

## Métricas de Evaluación
- **MSE (Mean Squared Error)**: Error cuadrático medio
- **Pixel Error**: Error en píxeles para reconstrucción de imágenes
- **Error máximo y promedio**: Estadísticas de error por píxel

## Visualizaciones
- Espacio latente con PCA
- Comparación de ruido y reconstrucción
- Distribución de errores
- Generación de muestras
- Caminos de interpolación

# Resultados

Los resultados se guardan en la carpeta `results/` e incluyen:
- Gráficos de entrenamiento
- Visualizaciones del espacio latente
- Comparaciones de reconstrucción
- Análisis de robustez al ruido
