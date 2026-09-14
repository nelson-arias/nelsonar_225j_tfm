import os
import argparse
import matplotlib.pyplot as plt
import csv
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd

from tf_pipeline import preprocess_image
from model import build_model, build_hybrid_model

def load_dataset_from_csv(csv_path):
    paths = []
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['image_path'] and row['label']:
                raw_path = row['image_path']
                raw_path = raw_path.replace('\\', '/')
                
                if 'words/' in raw_path:
                    rel_path = raw_path[raw_path.find('words/'):]
                    paths.append(rel_path)
                else:
                    paths.append(raw_path)
                    
                labels.append(int(row['label']))
    return paths, labels

def make_tf_dataset(paths, labels, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths))
        
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, augment=is_training), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_history(history, save_path="training_plot.png"):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

def save_image_samples(dataset, save_path, title, model=None):
    plt.figure(figsize=(15, 3))
    for images, labels in dataset.take(1):
        num_samples = min(5, len(images))
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        preds = None
        if model is not None:
            preds = model.predict(images, verbose=0)
            
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            true_label = labels[i].numpy()
            
            if preds is not None:
                pred_label = np.argmax(preds[i])
                confidence = np.max(preds[i]) * 100
                plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", fontsize=9)
            else:
                plt.title(f"Label: {true_label}")
            plt.axis('off')
            
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Imagen de muestras guardada en: {save_path}")

def train_and_evaluate_model(csv_path, model_type, prefix, args, external_test_dataset=None):
    print(f"\n{'='*60}")
    print(f"Iniciando entrenamiento: MODELO={model_type.upper()} | PREFIX={prefix.upper()}")
    print(f"{'='*60}")
    
    paths, labels = load_dataset_from_csv(csv_path)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    
    train_dataset = make_tf_dataset(train_paths, train_labels, batch_size=args.batch_size, is_training=True)
    val_dataset = make_tf_dataset(val_paths, val_labels, batch_size=args.batch_size, is_training=False)
    
    if external_test_dataset is not None:
        print("Utilizando test_dataset externo proporcionado para evaluación final.")
        test_dataset = external_test_dataset
    else:
        test_dataset = make_tf_dataset(test_paths, test_labels, batch_size=args.batch_size, is_training=False)

    save_image_samples(val_dataset, os.path.join(args.output_dir, f'{prefix}_raw_samples.png'), f"{prefix.capitalize()} - Muestras Crudas")
    save_image_samples(train_dataset, os.path.join(args.output_dir, f'{prefix}_augmented_samples.png'), f"{prefix.capitalize()} - Muestras Aumentadas")

    print(f"Construyendo arquitectura {model_type}..")
    model = build_model(model_type=model_type, num_classes=10)
    total_params = model.count_params()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    model_path = os.path.join(args.output_dir, f'{prefix}_model.keras')
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    elapsed_time = time.time() - start_time

    plot_path = os.path.join(args.output_dir, f'{prefix}_plot.png')
    plot_history(history, save_path=plot_path)
    
    gpus = tf.config.list_physical_devices('GPU')
    gpu_name = "CPU (No GPU detectada)"
    memory_peak = "N/A"
    
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get('device_name', 'GPU Genérica')
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            memory_peak_mb = mem_info['peak'] / (1024 * 1024)
            memory_peak = f"{memory_peak_mb:.2f} MB"
        except Exception as e:
            gpu_name = f"GPU Detectada (Error detalles: {e})"
            memory_peak = "Desconocido"

    epochs_run = len(history.history['loss'])
    time_per_epoch = elapsed_time / epochs_run if epochs_run > 0 else 0
    
    fig_stats, ax_stats = plt.subplots(figsize=(6, 4))
    ax_stats.axis('off')
    stats_text = (
        f"Estadísticas de Entrenamiento: {prefix.capitalize()}\n\n"
        f"GPU Utilizada: {gpu_name}\n"
        f"Memoria Peak (GPU): {memory_peak}\n"
        f"Tiempo Total Entrenamiento: {elapsed_time:.1f} s\n"
        f"Épocas Ejecutadas: {epochs_run}\n"
        f"Tiempo Medio por Época: {time_per_epoch:.2f} s/época"
    )
    ax_stats.text(0.5, 0.5, stats_text, fontsize=12, ha='center', va='center', 
                  bbox=dict(facecolor='#f0f0f0', alpha=0.8, boxstyle='round,pad=1'))
    plt.tight_layout()
    stats_path = os.path.join(args.output_dir, f'{prefix}_gpu_stats.png')
    plt.savefig(stats_path, dpi=150)
    plt.close()
    print(f"Estadísticas de GPU guardadas en: {stats_path}")
    
    print("Guardando predicciones y calculando métricas finales sobre Test Set...")
    model.load_weights(model_path)
    save_image_samples(test_dataset, os.path.join(args.output_dir, f'{prefix}_test_predictions.png'), f"{prefix.capitalize()} - Predicciones Test", model=model)
    
    all_preds = []
    all_true = []
    for images, lbls in test_dataset:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(lbls.numpy())
        
    f1 = f1_score(all_true, all_preds, average='weighted')
    precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_true, all_preds, average='weighted')
    
    test_loss, test_acc, test_top3 = model.evaluate(test_dataset, verbose=0)
    
    return {
        'model_type': model_type,
        'model_name': model.name,
        'parameters': total_params,
        'val_accuracy': test_acc,
        'val_loss': test_loss,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'training_time_sec': elapsed_time,
        'history': history,
        'test_dataset': test_dataset
    }

def generate_architectural_comparison_report(results_list, output_dir):
    print("\n" + "="*60)
    print("REPORTE COMPARATIVO DE ARQUITECTURAS (TOP 10 AUTORES - IAM DATASET)")
    print("="*60)
    
    df = pd.DataFrame([{
        'Arquitectura': res['model_name'],
        'Parámetros': f"{res['parameters']:,}",
        'Val Accuracy': f"{res['val_accuracy']:.4f}",
        'Val Loss': f"{res['val_loss']:.4f}",
        'F1 Score': f"{res['f1_score']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'Tiempo Total (s)': f"{res['training_time_sec']:.1f}"
    } for res in results_list])
    
    print(df.to_string(index=False))
    csv_path = os.path.join(output_dir, 'arch_comparison_metrics.csv')
    df.to_csv(csv_path, index=False)
    
    model_names = [res['model_name'] for res in results_list]
    metrics = ['Val Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    x = np.arange(len(model_names))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
    
    for idx, metric in enumerate(metrics):
        metric_key = metric.lower().replace(' ', '_')
        values = [res[metric_key] for res in results_list]
        rects = ax.bar(x + (idx - 1.5) * width, values, width, label=metric, color=colors[idx])
        ax.bar_label(rects, fmt='%.3f', padding=2, fontsize=8)
        
    ax.set_ylabel('Score')
    ax.set_title('Comparación de Desempeño por Arquitectura (Dataset Limpio Top 10 Autores)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, weight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'arch_comparison_metrics_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    curves_path = None
    has_histories = any(res.get('history') is not None for res in results_list)
    if has_histories:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        line_styles = ['-', '--', '-.']
        
        for idx, res in enumerate(results_list):
            if res.get('history') is None:
                continue
            h = res['history'].history if hasattr(res['history'], 'history') else res['history']
            epochs_range = range(len(h['val_sparse_categorical_accuracy']))
            name = res['model_name']
            style = line_styles[idx % len(line_styles)]
            
            ax1.plot(epochs_range, h['val_sparse_categorical_accuracy'], label=f"{name}", linestyle=style, linewidth=2)
            ax2.plot(epochs_range, h['val_loss'], label=f"{name}", linestyle=style, linewidth=2)
            
        ax1.set_title('Validation Accuracy en el Tiempo', fontsize=12, weight='bold')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        ax2.set_title('Validation Loss en el Tiempo', fontsize=12, weight='bold')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        curves_path = os.path.join(output_dir, 'arch_comparison_loss_curves.png')
        plt.savefig(curves_path, dpi=300)
        plt.close()

    fig_tbl, ax_tbl = plt.subplots(figsize=(12, 3 + len(df) * 0.5))
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    
    tbl = ax_tbl.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1F4E79')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#F2F4F4' if r % 2 == 0 else '#FFFFFF')
            
    plt.title("Tabla Comparativa Final de Arquitecturas - IAM Dataset Top 10", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    tbl_path = os.path.join(output_dir, 'arch_comparison_table.png')
    plt.savefig(tbl_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nArchivos comparativos generados exitosamente en '{output_dir}':")
    print(f"Métrica Barras: {plot_path}")
    if curves_path:
        print(f" - Curvas Aprendizaje: {curves_path}")
    print(f"Tabla Visual PNG: {tbl_path}")
    print(f"Datos CSV: {csv_path}")

def generate_comparison_report(clean_metrics, noisy_metrics, output_dir):
    print("\n" + "="*50)
    print("REPORTE COMPARATIVO CLEAN VS NOISY (PROPUESTA HÍBRIDA)")
    print("="*50)
    
    df = pd.DataFrame({
        'Métrica': ['Val Accuracy', 'Val Loss', 'F1 Score', 'Precision', 'Recall'],
        'Clean Model': [
            clean_metrics['val_accuracy'], clean_metrics['val_loss'], 
            clean_metrics['f1_score'], clean_metrics['precision'], clean_metrics['recall']
        ],
        'Noisy Model': [
            noisy_metrics['val_accuracy'], noisy_metrics['val_loss'], 
            noisy_metrics['f1_score'], noisy_metrics['precision'], noisy_metrics['recall']
        ]
    })
    
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, 'clean_vs_noisy_metrics.csv'), index=False)
    
    metrics = ['Val Accuracy', 'F1 Score', 'Precision', 'Recall']
    clean_vals = [clean_metrics['val_accuracy'], clean_metrics['f1_score'], clean_metrics['precision'], clean_metrics['recall']]
    noisy_vals = [noisy_metrics['val_accuracy'], noisy_metrics['f1_score'], noisy_metrics['precision'], noisy_metrics['recall']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, clean_vals, width, label='Clean Model', color='skyblue')
    rects2 = ax.bar(x + width/2, noisy_vals, width, label='Noisy Model', color='salmon')
    
    ax.set_ylabel('Scores')
    ax.set_title('Comparación Propuesta Híbrida: Clean vs Noisy')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.3f', padding=3)
    ax.bar_label(rects2, fmt='%.3f', padding=3)
    
    fig.tight_layout()
    comp_plot_path = os.path.join(output_dir, 'clean_vs_noisy_plot.png')
    plt.savefig(comp_plot_path)
    plt.close()
    
    print(f"\nReporte Clean vs Noisy guardado en: {comp_plot_path}")

def evaluate_saved_model(csv_path, model_type, prefix, output_dir, batch_size=32, external_test_dataset=None):
    model_path = os.path.join(output_dir, f'{prefix}_model.keras')
    print(f"\nEvaluando modelo guardado: {model_path} ({model_type})..")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo guardado: {model_path}")
        
    if external_test_dataset is not None:
        print("Utilizando test_dataset externo proporcionado para evaluación final.")
        test_dataset = external_test_dataset
    else:
        paths, labels = load_dataset_from_csv(csv_path)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=0.30, random_state=42, stratify=labels
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
        )
        test_dataset = make_tf_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False)
    
    print(f"Construyendo arquitectura '{model_type}' y cargando pesos..")
    model = build_model(model_type=model_type, num_classes=10)
    model.load_weights(model_path)
    total_params = model.count_params()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    print("Calculando evaluación en dataset de test..")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    
    all_preds = []
    all_true = []
    for images, lbls in test_dataset:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(lbls.numpy())
        
    f1 = f1_score(all_true, all_preds, average='weighted')
    precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_true, all_preds, average='weighted')
    
    return {
        'model_type': model_type,
        'model_name': model.name,
        'parameters': total_params,
        'val_accuracy': test_acc,
        'val_loss': test_loss,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'training_time_sec': 0.0,
        'history': None,
        'test_dataset': test_dataset
    }

def main():
    parser = argparse.ArgumentParser(description="Script de Entrenamiento y Comparación de Modelos IAM")
    parser.add_argument('--mode', type=str, default='compare_architectures', choices=['compare_architectures', 'clean_vs_noisy'], help='Modo de ejecución: compare_architectures (3 modelos) o clean_vs_noisy')
    parser.add_argument('--clean_csv', type=str, default='iam_top10_dataset.csv', help='Ruta al CSV limpio')
    parser.add_argument('--noisy_csv', type=str, default='iam_noisy_dataset.csv', help='Ruta al CSV con ruido')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote')
    parser.add_argument('--output_dir', type=str, default='./', help='Directorio para guardar modelos y gráficos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"GPUs disponibles: {tf.config.list_physical_devices('GPU')}")
    
    if args.mode == 'compare_architectures':
        architectures = [
            ('resnet_baseline', 'resnet_baseline'),
            ('crnn_bilstm', 'crnn_bilstm'),
            ('hybrid_transformer', 'hybrid_transformer')
        ]
        
        results = []
        for m_type, prefix in architectures:
            res = train_and_evaluate_model(args.clean_csv, model_type=m_type, prefix=prefix, args=args)
            results.append(res)
            
        generate_architectural_comparison_report(results, args.output_dir)

    elif args.mode == 'clean_vs_noisy':
        import shutil
        clean_path = os.path.join(args.output_dir, 'clean_hybrid_model.keras')
        prev_clean_path = os.path.join(args.output_dir, 'hybrid_transformer_model.keras')
        
        if not os.path.exists(clean_path) and os.path.exists(prev_clean_path):
            shutil.copyfile(prev_clean_path, clean_path)
            print(f"Reutilizando modelo limpio previa de la comparativa de arquitecturas: {prev_clean_path}")

        if os.path.exists(clean_path):
            print("Cargando y evaluando el modelo limpio pre-existente..")
            clean_metrics = evaluate_saved_model(args.clean_csv, model_type='hybrid_transformer', prefix='clean_hybrid', output_dir=args.output_dir, batch_size=args.batch_size)
        else:
            clean_metrics = train_and_evaluate_model(args.clean_csv, model_type='hybrid_transformer', prefix='clean_hybrid', args=args)
            
        test_dataset_limpio = clean_metrics['test_dataset']

        print("\nIniciando entrenamiento del modelo con Ruido (iam_noisy_dataset.csv)..")
        noisy_metrics = train_and_evaluate_model(args.noisy_csv, model_type='hybrid_transformer', prefix='noisy_hybrid', args=args, external_test_dataset=test_dataset_limpio)
        generate_comparison_report(clean_metrics, noisy_metrics, args.output_dir)

if __name__ == '__main__':
    main()
