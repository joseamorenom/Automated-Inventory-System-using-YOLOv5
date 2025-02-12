# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 00:05:47 2024

@author: amesa
"""

import os
from pathlib import Path
import pandas as pd
import cv2
import torch
from matplotlib import pyplot as plt

# Ruta al modelo entrenado y a las imágenes a probar
model_path = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\yolov5\runs\train\exp12\weights\best.pt"
images_dir = Path(r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\A_Prueba Final\vivoo")
output_dir = Path(r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\A_Prueba Final\vivoo\results")

# Clases del modelo entrenado
class_names = ['backpack', 'fan', 'podium', 'chair']

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Crear directorio de resultados
output_dir.mkdir(parents=True, exist_ok=True)

# Inventario final
inventory = []

# Procesar cada imagen
for image_path in images_dir.glob("*.jpg"):
    # Realizar predicción con YOLO
    results = model(str(image_path))
    
    # Obtener detecciones y contarlas
    detections = results.pandas().xyxy[0]  # DataFrame con detecciones
    counts = detections['name'].value_counts().to_dict()
    
    # Agregar información al inventario
    counts['image'] = image_path.stem
    inventory.append(counts)

    # Dibujar bounding boxes sobre la imagen
    img = cv2.imread(str(image_path))
    for _, row in detections.iterrows():
        # Coordenadas del bounding box
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        
        # Dibujar rectángulo y etiqueta
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Guardar la imagen con bounding boxes
    output_image_path = output_dir / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_image_path), img)

    # Mostrar imagen con bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecciones en {image_path.stem}")
    plt.axis('off')
    plt.show()

# Convertir el inventario a DataFrame y mostrarlo como tabla
inventory_df = pd.DataFrame(inventory).fillna(0)  # Rellenar NaN con 0
inventory_df = inventory_df.astype({cls: int for cls in class_names if cls in inventory_df.columns})
print("\nInventario final:")
print(inventory_df.to_string(index=False))

# Exportar inventario a CSV
inventory_csv_path = output_dir / "inventory_results.csv"
inventory_df.to_csv(inventory_csv_path, index=False)
print(f"\nInventario guardado en: {inventory_csv_path}")


#%%

import os
from pathlib import Path
import pandas as pd
import cv2
import torch
from matplotlib import pyplot as plt

# Ruta al modelo entrenado y a las imágenes a probar
model_path = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\yolov5\runs\train\exp12\weights\best.pt"
images_dir = Path(r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\A_Prueba Final\En_vivo")
output_dir = Path(r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\A_Prueba Final\En_vivo\results")

# Clases del modelo entrenado
class_names = ['backpack', 'fan', 'podium', 'chair']

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Crear directorio de resultados
output_dir.mkdir(parents=True, exist_ok=True)

# Inventario final
inventory = []

# Procesar cada imagen
for image_path in images_dir.glob("*.jpg"):
    # Realizar predicción con YOLO
    results = model(str(image_path))
    
    # Obtener detecciones y contarlas
    detections = results.pandas().xyxy[0]  # DataFrame con detecciones
    counts = detections['name'].value_counts().to_dict()
    
    # Agregar información al inventario
    counts['image'] = image_path.stem
    inventory.append(counts)

    # Dibujar bounding boxes sobre la imagen
    img = cv2.imread(str(image_path))
    for _, row in detections.iterrows():
        # Coordenadas del bounding box
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        
        # Dibujar rectángulo y etiqueta
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Guardar la imagen con bounding boxes
    output_image_path = output_dir / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_image_path), img)

    # Mostrar imagen con bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecciones en {image_path.stem}")
    plt.axis('off')
    plt.show()

# Convertir el inventario a DataFrame y mostrarlo como tabla
inventory_df = pd.DataFrame(inventory).fillna(0)  # Rellenar NaN con 0
inventory_df = inventory_df.astype({cls: int for cls in class_names if cls in inventory_df.columns})

# Asegurar que la columna 'image' sea la primera
columns_order = ['image'] + [col for col in inventory_df.columns if col != 'image']
inventory_df = inventory_df[columns_order]

print("\nInventario final:")
print(inventory_df.to_string(index=False))

# Exportar inventario a CSV
inventory_csv_path = output_dir / "inventory_results.csv"
inventory_df.to_csv(inventory_csv_path, index=False)
print(f"\nInventario guardado en: {inventory_csv_path}")