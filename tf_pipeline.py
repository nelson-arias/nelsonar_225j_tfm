import tensorflow as tf
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

IMG_HEIGHT = 64
IMG_WIDTH = 256

def preprocess_image(image_path_tensor, label_tensor, augment=False):
    def _process(path, label, aug_flag):
        path_str = path.decode('utf-8')
        
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        else:
            blur = cv2.GaussianBlur(img, (5,5), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            binary = cv2.bitwise_not(binary)
            
            if aug_flag:
                try:
                    import albumentations as A
                    transform = A.Compose([
                        A.ElasticTransform(alpha=25.0, sigma=5.0, alpha_affine=25.0, p=0.85),
                        A.OneOf([
                            A.Morphological(scale=(2, 2), operation='dilation', p=0.5),
                            A.Morphological(scale=(2, 2), operation='erosion', p=0.5),
                        ], p=0.7),
                        A.Affine(shear=(-15, 15), p=0.7)
                    ])
                    augmented = transform(image=binary)
                    binary = augmented['image']
                except ImportError:
                    pass

            h, w = binary.shape
            scale = IMG_HEIGHT / h
            new_w = int(w * scale)
            
            if new_w > IMG_WIDTH:
                new_w = IMG_WIDTH
                scale = IMG_WIDTH / w
                new_h = int(h * scale)
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
        
        return img, np.int64(label)

    image, label = tf.numpy_function(
        func=_process, 
        inp=[image_path_tensor, label_tensor, augment], 
        Tout=[tf.float32, tf.int64]
    )
    
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, 1))
    label.set_shape(())
    return image, label

def create_dataset(csv_path, batch_size=32, is_training=True):
    import csv
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

if __name__ == "__main__":
    csv_path = r"/content/drive/MyDrive/TFM_Handwriting/iam_top10_dataset.csv"
    
    print("Creando Dataset de validación para probar..")
    val_dataset = create_dataset(csv_path, batch_size=8, is_training=False)
    
    for images, labels in val_dataset.take(1):
        print("Shape del batch de imágenes:", images.shape)
        print("Etiquetas:", labels.numpy())
        
        plt.figure(figsize=(15, 6))
        for i in range(8):
            plt.subplot(2, 4, i+1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis('off')
        
        output_plot = "tf_dataset_sample.png"
        plt.savefig(output_plot)
        print(f"Lote de muestra guardado en.. {output_plot}")
        break
