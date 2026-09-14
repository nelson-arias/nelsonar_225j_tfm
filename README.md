# Identificación de Escritor de Texto Manuscrito en un Conjunto Cerrado de Autores Mediante Visión Artificial
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![GPU](https://img.shields.io/badge/Accelerated-NVIDIA%20A100-76B900.svg)](https://www.nvidia.com)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#)
Este repositorio contiene la implementación del Trabajo Fin de Máster (TFM) titulado:  
**«Identificación de escritor de texto manuscrito en un conjunto cerrado de autores mediante visión artificial»**  
*Máster Universitario en Formación permanente en Inteligencia Artificial — Universidad Europea de Valencia.*
---
## 📌 Descripción del Proyecto
El proyecto aborda la **identificación de autoría (*Writer Identification*)** a partir de imágenes de palabras manuscritas aisladas dentro de un escenario cerrado (*closed-set*) de 10 escritores seleccionados del benchmark estándar **IAM Handwriting Database**.
A diferencia del reconocimiento de texto tradicional (HTR/OCR) que decodifica el contenido semántico, este sistema extrae la huella biométrica y los rasgos caligráficos idiolectales del autor (morfología del trazo, inclinación y ligaduras) mediante una arquitectura híbrida de visión por computador y aprendizaje profundo:
1. **CNN Backbone:** Extracción jerárquica de características visuales y trazos mediante 5 bloques convolucionales con reducción espacial asimétrica (preservando el ancho).
2. **Capa Map-to-Sequence (M2S):** Transición matemática que colapsa la dimensión vertical para generar una secuencia espacial horizontal ordenada de izquierda a derecha.
3. **Transformer Encoder:** Módulo secuencial basado en mecanismos de auto-atención multicabeza (*Multi-Head Attention*) y *Positional Embeddings* para capturar dependencias de largo alcance en paralelo.
4. **Clasificador Final:** Pooling espacial global seguido de regularización y capa Softmax sobre las 10 clases cerradas.
Asimismo, se evalúa la propuesta frente a dos arquitecturas de referencia (**Baseline Convolucional Residual con GAP** y **Baseline Recurrente CRNN con BiLSTM**), se analiza la robustez ante ruido de etiquetas (*Data Poisoning* al 50%) y se incluye una prueba de concepto cualitativa para inferencia por mayoría en párrafos completos.
---
## 📁 Estructura del Repositorio
El código está estructurado de forma modular según los anexos metodológicos del proyecto:
```bash
├── iam_parser.py          # Parser XML/words.txt, filtrado de Top 10 y generación de CSVs
├── tf_pipeline.py         # Pipeline tf.data, binarización Otsu y Data Augmentation
├── model.py               # Definición de las 3 arquitecturas (ResNet, CRNN, Hybrid Transformer)
├── train.py               # Entrenamiento, partición 70/15/15, métricas y telemetría GPU
├── inference.py           # Prueba de concepto de inferencia en párrafos (segmentación OpenCV)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación del repositorio
```
⚙️ Requisitos e Instalación
Se recomienda utilizar un entorno virtual con Python 3.10 o superior y aceleración por GPU (CUDA):

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio

# Instalar dependencias
pip install -r requirements.txt
```
Contenido de requirements.txt:
text
tensorflow>=2.10.0
opencv-python>=4.7.0
albumentations>=1.3.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0

📊 Preparación del Dataset (IAM Handwriting Database)
Descargue el dataset desde el repositorio oficial de la Universidad de Berna (IAM).
Asegúrese de contar con la carpeta de imágenes words/, el archivo de transcripciones words.txt y la carpeta de metadatos xml/.
Ejecute el preprocesador para filtrar las palabras válidas del Top 10 de autores y generar los conjuntos de datos:
```bash
python iam_parser.py --base_dir ./iam_data --output_dir ./data_processed
```

Esto generará dos archivos:

**iam_top10_dataset.csv**: 4,322 palabras limpias de los 10 autores con más muestras.

**iam_noisy_dataset.csv**: Conjunto para el experimento de robustez (50% muestras legítimas + 50% muestras de autores externos con etiquetas corruptas).

🚀 Guía de Ejecución
1. Estudio Comparativo de Arquitecturas
Entrena secuencialmente las 3 redes bajo idénticas condiciones (partición estratificada 70% Train / 15% Validation / 15% Test independiente, optimizador Adam, paciencia de EarlyStopping 10 sobre val_loss):

```bash
python train.py \
  --mode compare_architectures \
  --clean_csv ./data_processed/iam_top10_dataset.csv \
  --epochs 100 \
  --batch_size 32 \
  --output_dir ./results_architectures
```

2. Evaluación de Robustez ante Ruido (Clean vs. Noisy)
Entrena la propuesta híbrida sobre el dataset corrupto y la evalúa rigurosamente sobre exactamente el mismo conjunto de test limpio:

```bash
python train.py \
  --mode clean_vs_noisy \
  --clean_csv ./data_processed/iam_top10_dataset.csv \
  --noisy_csv ./data_processed/iam_noisy_dataset.csv \
  --epochs 100 \
  --batch_size 32 \
  --output_dir ./results_noise
```

3. Prueba de Concepto: Inferencia en Párrafos
Ejecuta la segmentación morfológica por palabras con OpenCV y calcula la distribución de votos de autoría en una imagen de párrafo manuscrito:
```bash
python inference.py \
  --image ./inference_images/a01-000u.png \
  --model ./results_architectures/hybrid_transformer_best.keras \
  --output_dir ./inference_output
```

📈 Resultados Experimentales Principales
Comparativa en Conjunto de Prueba Independiente (Test Set: 649 muestras)
Entrenamientos ejecutados en GPU NVIDIA A100-SXM4-40GB (Google Colab Pro):

| Arquitectura | Parámetros | Test Accuracy | Test Loss | F1-Score | Épocas | Tiempo Medio / Época | Tiempo Total (s) | Aceleración |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet Baseline (Adaptada + GAP)** | 2,462,986 | 94.62% | 0.1871 | 94.61% | 55 | 32.62 s | 1,793.9 s (29.9 min) | Baseline |
| **CRNN Baseline (BiLSTM)** | 3,611,530 | 96.12% | 0.1374 | 96.09% | 61 | 29.73 s | 1,813.3 s (30.2 min) | 1.02x |
| **Propuesta Híbrida (CNN + Transformer)** | 3,631,498 | 94.47% | 0.1750 | 94.43% | 39 | 31.00 s | 1,209.2 s (20.0 min) | 1.50x (33.7% más rápido) |


Hallazgo Clave de Eficiencia: La latencia computacional por época es prácticamente equivalente entre CRNN y Transformer (~30 s/época). La reducción del 33.7% en el tiempo total obedece a una mayor tasa de convergencia algorítmica de la auto-atención, requiriendo un 56.4% menos de épocas (39 vs. 61) para estabilizarse bajo el mismo criterio de parada temprana.

Evaluación de Robustez ante Ruido (Data Poisoning al 50%)
Evaluados sobre el mismo conjunto de test limpio:

Modelo Limpio: 94.47% Accuracy | 0.1750 Loss | 94.43% F1-Score
Modelo con Ruido: 91.86% Accuracy (-2.88%) | 0.3695 Loss (+111% incremento) | 91.81% F1-Score

⚖️ Delimitación Metodológica y Alcance

**Escenario Cerrado (Closed-Set)**: El sistema está estrictamente acotado a clasificar palabras aisladas entre los 10 autores registrados de entrenamiento.

**Inferencia en Párrafos**: Se incluye como una prueba de concepto preliminar de agregación de predicciones por mayoría. No constituye un sistema de diarización multi-autor formalmente validado ni un mecanismo probabilístico fiable para la detección o rechazo de autores desconocidos (open-set), aspectos delimitados como líneas futuras de investigación.

🎓 Autor y Agradecimientos

**Autor**: Nelson Mauricio Arias.

**Titulación**: Máster Universitario en Formación permanente en Inteligencia Artificial

**Institución**: Universidad Europea de Valencia

**Agradecimientos**: A mi tutor y al Instituto de Informática de la Universidad de Berna por facilitar el dataset benchmark IAM Handwriting Database.
