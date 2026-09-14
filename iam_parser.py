import os
import glob
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
import random

def parse_iam_metadata(base_dir, top_n=10):
    xml_dir = os.path.join(base_dir, 'xml')
    words_txt_path = os.path.join(base_dir, 'words.txt')
    
    print("1. Parsing XML files to map form_id -> writer_id...")
    form_to_writer = {}
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            form_id = root.attrib.get('id')
            writer_id = root.attrib.get('writer-id')
            if form_id and writer_id:
                form_to_writer[form_id] = writer_id
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            
    print(f"Found {len(form_to_writer)} forms with writer IDs.")
    
    print("2. Parsing words.txt to find valid words...")
    writer_word_counts = defaultdict(int)
    valid_words = []
    
    with open(words_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                word_id = parts[0]
                status = parts[1]
                
                if status == 'ok':
                    # Extract form ID from word ID (e.g., a01-000u-00-00 -> a01-000u)
                    form_id = '-'.join(word_id.split('-')[:2])
                    writer_id = form_to_writer.get(form_id)
                    
                    if writer_id:
                        writer_word_counts[writer_id] += 1
                        valid_words.append({
                            'word_id': word_id,
                            'form_id': form_id,
                            'writer_id': writer_id,
                            'transcription': parts[-1]
                        })
    
    print(f"Found {len(valid_words)} valid 'ok' words.")
    
    print("3. Selecting Top N authors...")
    sorted_writers = sorted(writer_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_writers = sorted_writers[:top_n]
    
    top_writer_ids = [w[0] for w in top_writers]
    print(f"Top {top_n} writers:")
    for w_id, count in top_writers:
        print(f"  Writer {w_id}: {count} words")
        
    print("4. Creating dataset mapping...")
    # Map top writer IDs to labels 0-9
    writer_to_label = {w_id: idx for idx, w_id in enumerate(top_writer_ids)}
    
    dataset_records = []
    words_dir = os.path.join(base_dir, 'words')
    
    for word_info in valid_words:
        w_id = word_info['writer_id']
        if w_id in top_writer_ids:
            word_id = word_info['word_id']
            # Image path logic: words/a01/a01-000u/a01-000u-00-00.png
            parts = word_id.split('-')
            dir1 = parts[0]
            dir2 = f"{parts[0]}-{parts[1]}"
            img_name = f"{word_id}.png"
            img_path = os.path.join(words_dir, dir1, dir2, img_name)
            
            dataset_records.append({
                'image_path': img_path,
                'author_id': w_id,
                'label': writer_to_label[w_id]
            })
            
    # Guardar Clean Dataset
    output_clean_csv = os.path.join(os.path.dirname(base_dir), 'iam_top10_dataset.csv')
    with open(output_clean_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'author_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset_records:
            writer.writerow(record)
            
    print(f"Successfully saved clean dataset to: {output_clean_csv}")
    print(f"Total images in clean dataset: {len(dataset_records)}")

    # ---------------- GENERAR NOISY DATASET ----------------
    print("\n5. Creating noisy dataset (exclusively from clean training partition)...")
    from sklearn.model_selection import train_test_split
    
    # Particionar el dataset limpio de forma estratificada e idéntica a train.py (70% train, 15% val, 15% test)
    clean_labels = [r['label'] for r in dataset_records]
    train_records, temp_records, _, temp_labels = train_test_split(
        dataset_records, clean_labels, test_size=0.30, random_state=42, stratify=clean_labels
    )
    val_records, test_records, _, _ = train_test_split(
        temp_records, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    
    print(f"Particiones del dataset limpio: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print("Garantía metodológica: Las particiones de Validación y Test quedan 100% aisladas, limpias y excluidas del entrenamiento ruidoso.")
    
    # El 50% de datos limpios se toma EXCLUSIVAMENTE de la partición de entrenamiento (Train)
    random.seed(42)
    num_clean_needed = len(train_records) // 2
    shuffled_clean_train = list(train_records)
    random.shuffle(shuffled_clean_train)
    noisy_dataset_records = shuffled_clean_train[:num_clean_needed]
    
    # El 50% restante: extraer imágenes de autores del 11 en adelante
    noise_writers = sorted_writers[top_n:]
    noise_writer_ids = [w[0] for w in noise_writers]
    
    all_noise_records = []
    for word_info in valid_words:
        w_id = word_info['writer_id']
        if w_id in noise_writer_ids:
            word_id = word_info['word_id']
            parts = word_id.split('-')
            dir1 = parts[0]
            dir2 = f"{parts[0]}-{parts[1]}"
            img_name = f"{word_id}.png"
            img_path = os.path.join(words_dir, dir1, dir2, img_name)
            
            # Etiqueta falsa para meter ruido en las 10 clases
            fake_label = random.randint(0, top_n - 1)
            
            all_noise_records.append({
                'image_path': img_path,
                'author_id': w_id,
                'label': fake_label
            })
            
    # Seleccionar la cantidad necesaria de ruido para completar el tamaño de entrenamiento
    random.shuffle(all_noise_records)
    num_noise_needed = len(train_records) - num_clean_needed
    
    if len(all_noise_records) < num_noise_needed:
        print(f"Warning: Not enough noise records ({len(all_noise_records)} available, {num_noise_needed} needed). Taking all available.")
        selected_noise = all_noise_records
    else:
        selected_noise = all_noise_records[:num_noise_needed]
        
    noisy_dataset_records.extend(selected_noise)
    random.shuffle(noisy_dataset_records) # Mezclar limpio y ruido
    
    output_noisy_csv = os.path.join(os.path.dirname(base_dir), 'iam_noisy_dataset.csv')
    with open(output_noisy_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'author_id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in noisy_dataset_records:
            writer.writerow(record)
            
    print(f"Successfully saved noisy dataset to: {output_noisy_csv}")
    print(f"Total imágenes en el conjunto ruidoso de entrenamiento: {len(noisy_dataset_records)} (Limpias de Train: {num_clean_needed}, Ruido externo: {len(selected_noise)})")


if __name__ == "__main__":
    #base_dir = r"E:\Shared_folder\MASTER_IA_UEV_DOCS\TFM\datasets\IAM Handwriting Database"
    base_dir = r"/content/drive/MyDrive/TFM_Handwriting"
    parse_iam_metadata(base_dir, top_n=10)
