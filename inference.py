import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os
from collections import Counter
from model import build_hybrid_model

IMG_HEIGHT = 64
IMG_WIDTH = 256

def preprocess_for_inference(word_img):
    if word_img is None or word_img.size == 0:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    # Convertir a escala de grises si tiene 3 canales
    if len(word_img.shape) == 3:
        word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        
    h, w = word_img.shape
    if h == 0 or w == 0:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        
    _, binary = cv2.threshold(word_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir si es necesario (el fondo debe ser negro y el trazo blanco)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        
    scale = IMG_HEIGHT / max(1, h)
    new_w = max(1, int(w * scale))
    
    if new_w > IMG_WIDTH:
        new_w = IMG_WIDTH
        scale = IMG_WIDTH / max(1, w)
        new_h = max(1, int(h * scale))
        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_top = (IMG_HEIGHT - new_h) // 2
        pad_bottom = IMG_HEIGHT - new_h - pad_top
        img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        resized = cv2.resize(binary, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        pad_right = IMG_WIDTH - new_w
        img = cv2.copyMakeBorder(resized, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
        
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def segment_words(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarización invertida para que el texto sea blanco y el fondo negro
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilatación vertical/horizontal para conectar letras en palabras
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Encontrar contornos de las palabras
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    words_bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtrar contornos demasiado pequeños o estrechos (ruido)
        if w >= 10 and h >= 10 and (w * h) >= 100:
            words_bboxes.append((x, y, w, h))
            
    # Ordenar las palabras de arriba hacia abajo y de izquierda a derecha (aproximado)
    words_bboxes = sorted(words_bboxes, key=lambda b: (b[1] // 30, b[0]))
    return img, words_bboxes

def run_inference(image_path, model_path, output_dir):
    print(f"Cargando modelo desde {model_path}...")
    model = build_hybrid_model(num_classes=10)
    model.load_weights(model_path)
    
    print(f"Segmentando palabras de {image_path}...")
    original_img, bboxes = segment_words(image_path)
    
    if not bboxes:
        print("No se encontraron palabras en la imagen.")
        return
        
    print(f"Se encontraron {len(bboxes)} palabras.")
    
    predictions = []
    tensors = []
    
    for (x, y, w, h) in bboxes:
        word_crop = original_img[y:y+h, x:x+w]
        tensor = preprocess_for_inference(word_crop)
        tensors.append(tensor)
        
    tensors_batch = np.array(tensors)
    print("Ejecutando inferencia en batch...")
    preds = model.predict(tensors_batch)
    predicted_labels = np.argmax(preds, axis=1)
    
    # Dibujar resultados en la imagen
    result_img = original_img.copy()
    for (x, y, w, h), label in zip(bboxes, predicted_labels):
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, f"A{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    result_img_path = os.path.join(output_dir, 'inference_result.png')
    cv2.imwrite(result_img_path, result_img)
    print(f"Imagen con bounding boxes y predicciones guardada en {result_img_path}")
    
    # Generar gráficos
    author_counts = Counter(predicted_labels)
    labels = [f"Autor {a}" for a in author_counts.keys()]
    sizes = list(author_counts.values())
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Autores (Diarization)')
    
    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color='skyblue')
    plt.title('Palabras por Autor')
    plt.ylabel('Cantidad')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'inference_chart.png')
    plt.savefig(chart_path)
    print(f"Gráficos guardados en {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Inferencia para Párrafos")
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen del párrafo')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo entrenado (.keras)')
    parser.add_argument('--output_dir', type=str, default='./', help='Directorio para guardar los resultados')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_inference(args.image, args.model, args.output_dir)
