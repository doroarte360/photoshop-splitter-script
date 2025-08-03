import cv2
import os

def detect_larger_images(image_path, min_area=50000, overlap_threshold=0.5):
    """
    Detecta las regiones más grandes dentro de una imagen y omite áreas pequeñas o solapadas.

    Args:
        image_path (str): Ruta de la imagen a procesar.
        min_area (int): Área mínima para considerar un contorno como válido.
        overlap_threshold (float): Porcentaje de solapamiento permitido entre regiones.

    Returns:
        list: Lista de coordenadas [(x, y, w, h), ...] para cada región detectada.
        numpy.ndarray: Imagen original cargada.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suavizar la imagen para eliminar ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por tamaño
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area:
            regions.append((x, y, w, h))

    # Filtrar regiones solapadas
    regions = filter_overlapping_regions(regions, overlap_threshold)

    return regions, image

def filter_overlapping_regions(regions, threshold):
    """
    Filtra las regiones para eliminar las que están contenidas dentro de otras.

    Args:
        regions (list): Lista de coordenadas [(x, y, w, h)].
        threshold (float): Porcentaje de solapamiento permitido.

    Returns:
        list: Lista de regiones filtradas.
    """
    filtered_regions = []
    for i, (x1, y1, w1, h1) in enumerate(regions):
        overlap = False
        for j, (x2, y2, w2, h2) in enumerate(regions):
            if i != j:
                # Calcular solapamiento
                xa = max(x1, x2)
                ya = max(y1, y2)
                xb = min(x1 + w1, x2 + w2)
                yb = min(y1 + h1, y2 + h2)
                overlap_area = max(0, xb - xa) * max(0, yb - ya)
                region_area = w1 * h1
                if overlap_area / region_area > threshold:
                    overlap = True
                    break
        if not overlap:
            filtered_regions.append((x1, y1, w1, h1))
    return filtered_regions

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
min_area = 50000                       # Área mínima para detectar regiones grandes
overlap_threshold = 0.5                # Umbral para eliminar regiones solapadas

# Procesar la imagen
regions, original_image = detect_larger_images(image_path, min_area, overlap_threshold)

# Visualizar las regiones detectadas (opcional)
for (x, y, w, h) in regions:
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detecciones", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar las regiones detectadas
save_detected_regions(original_image, regions, output_dir)
print(f"Regiones detectadas: {len(regions)}")
