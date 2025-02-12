# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:24:56 2024

@author: amesa
"""

#%% Instalo las dependencias de YOLO

#pip install torch torchvision torchaudio
#pip install git+https://github.com/ultralytics/yolov5.git

#%%Data augmentation

import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# Directorios de imágenes y etiquetas
carpeta_imagenes = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\Chair - Copy"
carpeta_etiquetas = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\TagsChairCopia"
carpeta_aumentadas = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\prueba"

# Transformaciones para data augmentation
transform = A.Compose([
    A.RandomRotate90(p=0.5),          # Rotación aleatoria de 90 grados
    A.HorizontalFlip(p=0.5),          # Volteo horizontal
    A.VerticalFlip(p=0.5),            # Volteo vertical
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5)  # Desplazamiento, escala y rotación
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Función para aplicar data augmentation
def augment_images():
    if not os.path.exists(carpeta_aumentadas):
        os.makedirs(carpeta_aumentadas)

    for img_name in tqdm(os.listdir(carpeta_imagenes)):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(carpeta_imagenes, img_name)
            label_path = os.path.join(carpeta_etiquetas, img_name.replace(".jpg", ".txt"))

            # Leer imagen y etiquetas
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()

            bboxes = []
            class_labels = []
            for line in lines:
                label = line.strip().split()
                class_id = int(label[0])
                x_center, y_center, box_width, box_height = map(float, label[1:])
                bboxes.append([x_center, y_center, box_width, box_height])
                class_labels.append(class_id)

            # Aplicar transformaciones
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            # Guardar imagen aumentada
            aug_img_name = f"aug_{img_name}"
            aug_img_path = os.path.join(carpeta_aumentadas, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)

            # Guardar etiquetas aumentadas
            aug_label_path = os.path.join(carpeta_aumentadas, aug_img_name.replace(".jpg", ".txt"))
            with open(aug_label_path, 'w') as f:
                for bbox, class_label in zip(aug_bboxes, aug_class_labels):
                    f.write(f"{class_label} {' '.join(map(str, bbox))}\n")

augment_images()



#%% Redimensiono
import os
import cv2
import albumentations as A
from tqdm import tqdm

# Rutas de imágenes y etiquetas
carpeta_imagenes = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\Single\Full"
carpeta_etiquetas = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\Single\Full\tags"
carpeta_salida_imagenes = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\Single\FullRe"
carpeta_salida_etiquetas = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\Single\tagsRe"

# Crear carpetas de salida si no existen
os.makedirs(carpeta_salida_imagenes, exist_ok=True)
os.makedirs(carpeta_salida_etiquetas, exist_ok=True)

# Transformación para redimensionar y ajustar bounding boxes
transform = A.Compose([
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Función para redimensionar imágenes y ajustar etiquetas
def redimensionar_y_ajustar_etiquetas():
    for img_name in tqdm(os.listdir(carpeta_imagenes)):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(carpeta_imagenes, img_name)
            label_path = os.path.join(carpeta_etiquetas, img_name.replace(".jpg", ".txt"))

            # Verificar si existe el archivo de etiquetas
            if not os.path.exists(label_path):
                print(f"Archivo de etiquetas no encontrado: {label_path}")
                continue  # Saltar esta imagen si no hay etiquetas

            # Leer imagen y etiquetas
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()

            bboxes = []
            class_labels = []
            for line in lines:
                label = line.strip().split()
                class_id = int(float(label[0]))  # Convertir a entero
                x_center, y_center, box_width, box_height = map(float, label[1:])
                bboxes.append([x_center, y_center, box_width, box_height])
                class_labels.append(class_id)

            # Aplicar redimensionamiento y ajustar bounding boxes
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_class_labels = transformed['class_labels']

            # Guardar la imagen redimensionada
            cv2.imwrite(os.path.join(carpeta_salida_imagenes, img_name), aug_img)

            # Guardar las etiquetas ajustadas
            salida_etiqueta_path = os.path.join(carpeta_salida_etiquetas, img_name.replace(".jpg", ".txt"))
            with open(salida_etiqueta_path, 'w') as f_out:
                for bbox, class_label in zip(aug_bboxes, aug_class_labels):
                    f_out.write(f"{int(class_label)} {' '.join(map(str, bbox))}\n")  # Guardar clase como entero

redimensionar_y_ajustar_etiquetas()

#%% Reorganizo las fotos para el processed dataset
import os
import shutil
from sklearn.model_selection import train_test_split

# Directorio base donde están las carpetas de las clases
base_dir = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\Single"  # Cambia a la ruta de tu carpeta 'Single'

# Nombres de las carpetas de cada clase
class_dirs = ["Backpacks", "Chair", "Fan", "Podium", "Full"]

# Directorio de salida para la estructura organizada
output_dir = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\AA-Prueba Final\processed_dataset"  # Cambia esta ruta si deseas
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

# Iterar sobre cada carpeta de clase para organizar imágenes y etiquetas
for class_name in class_dirs:
    img_dir = os.path.join(base_dir, class_name)
    tag_dir = os.path.join(img_dir, "Tags")  # Carpeta que contiene las etiquetas
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Dividir en 80% train y 20% val
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Mover las imágenes y etiquetas de entrenamiento y validación
    for img_set, folder in zip([train_imgs, val_imgs], ["train", "val"]):
        for img in img_set:
            # Mover imagen
            shutil.copy(os.path.join(img_dir, img), os.path.join(output_dir, f"images/{folder}", img))

            # Mover etiqueta correspondiente
            label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(tag_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, f"labels/{folder}", label_file))

print("Organización completada.")

#%% Codigo para entrenar



#%%Prueba e YOLO
python train.py --img 640 --batch 16 --epochs 10 --data r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\Prueba YOLO\dataset.yaml" --weights yolov5s.pt


#%%Test del modelo resultante
#despues de correr esta linea:
    #python detect.py --weights "C:/Users/amesa/OneDrive/Universidad/Fundamentos/Images/yolov5/runs/train/exp11/weights/best.pt" --source "C:/Users/amesa/OneDrive/Universidad/Fundamentos/Images/Prueba Full room" --img 640 --conf 0.25

import os
import pandas as pd
from pathlib import Path

# Ruta a los resultados de detección
results_dir = Path(r'C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\yolov5\runs\detect\exp')
inventory = []

# Procesar cada archivo de detección en el directorio de resultados
for result_file in results_dir.glob('*.txt'):
    with open(result_file, 'r') as file:
        detections = file.readlines()
        counts = {}
        for line in detections:
            class_id = int(line.split()[0])
            # Convertir id de clase a nombre
            class_name = ['backpack', 'fan', 'podium', 'chair'][class_id]
            counts[class_name] = counts.get(class_name, 0) + 1
        counts['image'] = result_file.stem  # Nombre de la imagen
        inventory.append(counts)

# Convertir a DataFrame y exportar a CSV
inventory_df = pd.DataFrame(inventory).fillna(0)  # Rellena NaN con 0 para clases no detectadas
inventory_df.to_csv('inventory_results.csv', index=False)
print("Inventario guardado en 'inventory_results.csv'")

