import cv2
import numpy as np
import os


def detect_and_save_photos(page_path, output_dir, min_area=1000, margin=0):
    # Cargar la imagen
    image = cv2.imread(page_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {page_path}")

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ajustar el contraste para resaltar los bordes
    gray = cv2.equalizeHist(gray)

    # Binarización adaptativa para detectar bordes claros
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilatar para reforzar los bordes y conectar líneas fragmentadas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Procesar contornos detectados
    photo_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h  # Relación de aspecto

        # Filtrar regiones pequeñas y asegurar que las formas sean rectangulares (fotos completas)
        if area >= min_area and 0.8 < aspect_ratio < 1.2:  # Se ajusta a fotos cuadradas o casi rectangulares
            # Recortar la región detectada
            cropped_image = image[y:y + h, x:x + w]

            # Guardar la región como un archivo independiente
            photo_count += 1
            photo_path = os.path.join(output_dir, f"photo_{photo_count}.jpg")
            cv2.imwrite(photo_path, cropped_image)

    print(f"Se han guardado {photo_count} fotos completas en la carpeta: {output_dir}")


# Rutas de entrada y salida
page_path = r"C:/Users/tonip/Pictures/testphop/lote/Pagi0n117.jpg"
output_dir = r"C:/Users/tonip/Pictures/testphop/restaurada"

# Ejecutar el procesamiento
detect_and_save_photos(page_path, output_dir)
