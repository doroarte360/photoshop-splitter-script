import cv2
import numpy as np


def detect_photos(page_path, min_area=1000, margin=5):
    # Cargar la imagen
    image = cv2.imread(page_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {page_path}")

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ajustar el contraste
    gray = cv2.equalizeHist(gray)

    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área y forma
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h  # Relación de aspecto
        if area >= min_area and 0.5 < aspect_ratio < 2.0:  # Filtrar por tamaño y proporción
            # Expandir el contorno para incluir márgenes
            x = max(0, x - margin)
            y = max(0, y - margin)
            w += 2 * margin
            h += 2 * margin
            regions.append((x, y, w, h))

    # Ordenar regiones de arriba hacia abajo y de izquierda a derecha
    regions = sorted(regions, key=lambda r: (r[1], r[0]))

    return regions, image


# Probar con la imagen subida
page_path = "C:/Users/tonip/Pictures/testphop/Pagi0n121.jpg"
regions, image = detect_photos(page_path)

# Dibujar los contornos detectados
for x, y, w, h in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Guardar y mostrar resultado
output_path = "C:/Users/tonip/Pictures/testphop"
cv2.imwrite(output_path, image)
print(f"Imagen procesada guardada en: {output_path}")
