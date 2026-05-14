import cv2
import matplotlib.pyplot as plt
import os
import random
import matplotlib.patches as patches
import numpy as np 
def visualizar_etiquetas_yolo(ruta_imagen, ruta_etiqueta):
    print("ruta_imagen: ",ruta_imagen)
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró la imagen en {ruta_imagen}")
        return
    
    # Cargar imagen
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print(f"Error: No se pudo leer la imagen")
        return
        
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    alto_img, ancho_img, _ = imagen_rgb.shape

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(imagen_rgb)
    ax.set_title(f"Visualizando Etiquetado: {os.path.basename(ruta_imagen)}")
    ax.axis('off')

    # Leer etiquetas
    with open(ruta_etiqueta, 'r') as f:
        lineas = f.readlines()
        if not lineas:
            print(f"El archivo {ruta_etiqueta} está vacío.")
            plt.close()
            return

    for linea in lineas:
        partes = linea.strip().split()
        if len(partes) < 5: continue

        # YOLO: class_id x_center y_center width height
        x_c, y_c, w_n, h_n = map(float, partes[1:5])

        # Convertir coordenadas normalizadas a píxeles
        ancho_box = w_n * ancho_img
        alto_box = h_n * alto_img
        x1 = (x_c * ancho_img) - (ancho_box / 2)
        y1 = (y_c * alto_img) - (alto_box / 2)

        rect = patches.Rectangle((x1, y1), ancho_box, alto_box, 
                                 linewidth=2, edgecolor='#00FF00', facecolor='none')
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    base_path = r'C:\Users\cripe\Desktop\GANADO'
    
    # Definimos rutas
    images_path = os.path.join(base_path, 'images', 'val')
    labels_path = os.path.join(base_path, 'labels', 'val')

    # Validar que existan las carpetas
    if not os.path.exists(labels_path):
        print(f"No se encuentra la carpeta de etiquetas: {labels_path}")
    else:
        # LISTAMOS LOS ETIQUETADOS (.txt) primero para asegurar que solo abrimos lo que tiene etiquetas
        etiquetas = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        
        if etiquetas:
            # Elegir un archivo de etiqueta al azar
            txt_elegido = random.choice(etiquetas)
            nombre_base = os.path.splitext(txt_elegido)[0]
            
            # Buscar la imagen correspondiente (probando extensiones comunes)
            imagen_correspondiente = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                posible_img = os.path.join(images_path, nombre_base + ext)
                if os.path.exists(posible_img):
                    imagen_correspondiente = posible_img
                    break
            
            if imagen_correspondiente:
                print(f"Abriendo archivo etiquetado: {txt_elegido}")
                data = np.loadtxt( os.path.join(labels_path, txt_elegido))
                count =0
                while len(data)==0 and count<500:
                    count +=1

                    txt_elegido = random.choice(etiquetas)
                    nombre_base = os.path.splitext(txt_elegido)[0]

                    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                        posible_img = os.path.join(images_path, nombre_base + ext)
                        if os.path.exists(posible_img):
                            imagen_correspondiente = posible_img
                            break
                    data = np.loadtxt( os.path.join(labels_path, txt_elegido)) 
                visualizar_etiquetas_yolo(imagen_correspondiente, os.path.join(labels_path, txt_elegido))
            else:
                print(f"Se encontró la etiqueta {txt_elegido} pero no su imagen en {images_path}")
        else:
            print(f"No se encontraron archivos .txt en {labels_path}")