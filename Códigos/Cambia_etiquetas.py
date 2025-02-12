# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:01:13 2024

@author: amesa
"""

#%% Para arreglar lo de las clases

import os

# Ruta de la carpeta que contiene los archivos de etiquetas (.txt)
ruta_carpeta = r"C:\Users\amesa\OneDrive\Universidad\Fundamentos\Images\Fulroom_ya\tags"

# Diccionario de mapeo: clases viejas a clases nuevas
mapa_clases = {
    "15": "0",  # Cambiar clase 15 a 0 (backpack)
    "16": "1",  # Cambiar clase 16 a 1 (fan)
    "17": "2",  # Cambiar clase 17 a 2 (podium)
    "18": "3"   # Cambiar clase 18 a 3 (chair)
}

# Recorrer todos los archivos de la carpeta
for archivo in os.listdir(ruta_carpeta):
    if archivo.endswith(".txt"):  # Filtrar solo los archivos .txt
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        
        # Leer el contenido del archivo
        with open(ruta_archivo, 'r') as f:
            lineas = f.readlines()
        
        # Reemplazar la primera clase en cada línea
        nuevas_lineas = []
        for linea in lineas:
            partes = linea.split()  # Dividir la línea en partes (clase, x_center, y_center, etc.)
            clase = partes[0]       # Tomar la clase actual
            if clase in mapa_clases:  # Si la clase está en el diccionario de mapeo
                partes[0] = mapa_clases[clase]  # Cambiar la clase según el diccionario
            nuevas_lineas.append(" ".join(partes) + "\n")  # Volver a juntar la línea
        
        # Escribir las líneas actualizadas de vuelta al archivo
        with open(ruta_archivo, 'w') as f:
            f.writelines(nuevas_lineas)

print("Clases actualizadas correctamente.")


