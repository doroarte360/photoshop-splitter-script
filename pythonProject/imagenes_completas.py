import cv2
import os

def detect_images(image_path, min_area=50000):
    """
    Detecta imágenes completas en un archivo conteniendo varias imágenes.

    Args:
        image_path (str): Ruta de la imagen que contiene múltiples subimágenes.
        min_area (int): Área mínima para considerar un contorno como una imagen.

    Returns:
        list: Lista de coordenadas [(x, y, w, h), ...] para cada región detectada.
        numpy.ndarray: Imagen original cargada.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarizar para distinguir claramente las áreas
    _, binary = cv2.threshold(gray, 208, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por tamaño
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area:
            regions.append((x, y, w, h))

    return regions, image

def save_detected_regions(image, regions, output_dir):
    """
    Guarda las regiones detectadas como archivos de imagen separados.

    Args:
        image (numpy.ndarray): Imagen original.
        regions (list): Lista de coordenadas [(x, y, w, h)].
        output_dir (str): Directorio donde se guardarán las imágenes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x, y, w, h) in enumerate(regions):
        cropped = image[y:y + h, x:x + w]
        output_path = os.path.join(output_dir, f"imagen_{i + 1}.jpg")
        cv2.imwrite(output_path, cropped)
        print(f"Guardado: {output_path}")

# Parámetros
image_path = "C:/Users/tonip/Pictures/testphop/lote/Pagi0n117.jpg"  # Cambia esta ruta
output_dir = "C:/Users/tonip/Pictures/testphop/restaurada"       # Cambia esta ruta
min_area = 50000                       # Área mínima para detectar imágenes completas

# Procesar la imagen
regions, original_image = detect_images(image_path, min_area)

# Visualizar las regiones detectadas
for (x, y, w, h) in regions:
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detecciones", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar las regiones detectadas
save_detected_regions(original_image, regions, output_dir)
print(f"Regiones detectadas: {len(regions)}")
