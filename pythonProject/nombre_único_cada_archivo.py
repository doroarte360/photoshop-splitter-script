import cv2
import os
from datetime import datetime

# Variables globales
drawing = False  # Si se está dibujando un rectángulo
start_x, start_y = -1, -1  # Coordenadas iniciales para dibujar
rectangles = []  # Lista de rectángulos creados
image_resized = None  # Imagen redimensionada para vista previa
image = None  # Imagen original (sin redimensionar)


def create_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, rectangles, image, image_resized

    if event == cv2.EVENT_LBUTTONDOWN:  # Inicio del dibujo
        drawing = True
        start_x, start_y = x, y  # Guardar las coordenadas iniciales

    elif event == cv2.EVENT_MOUSEMOVE:  # Durante el movimiento del mouse
        if drawing:  # Si estamos dibujando
            img_copy = image_resized.copy()  # Copiar la imagen redimensionada para evitar sobrescribir
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Vista previa", img_copy)  # Mostrar la imagen con el rectángulo en proceso

    elif event == cv2.EVENT_LBUTTONUP:  # Finalización del dibujo
        drawing = False
        # Calcular las coordenadas del rectángulo en la imagen original
        ratio_x = image.shape[1] / image_resized.shape[1]
        ratio_y = image.shape[0] / image_resized.shape[0]
        rect_x = int(start_x * ratio_x)
        rect_y = int(start_y * ratio_y)
        rect_w = int((x - start_x) * ratio_x)
        rect_h = int((y - start_y) * ratio_y)
        rectangles.append((rect_x, rect_y, rect_w, rect_h))  # Añadir el rectángulo a la lista


def save_detected_regions(image, rectangles, output_dir):
    """Guardar las regiones detectadas como archivos de imagen separados"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x, y, w, h) in enumerate(rectangles):
        cropped = image[y:y + h, x:x + w]

        # Generar un nombre único para el archivo basado en la hora y fecha
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_filename = f"imagen_{timestamp}_{i + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Guardar la imagen
        cv2.imwrite(output_path, cropped)
        print(f"Guardado: {output_path}")


# Parámetros
image_path = "E:/gruyter/ajustes nuevos/albumviaje/img20240625_11520329.jpg"  # Cambia esta ruta
output_dir = "E:/gruyter/ajustes nuevos/restaurar_viaje/restaurada_rojo"  # Cambia esta ruta

# Cargar la imagen
image = cv2.imread(image_path)

# Redimensionar la imagen al 30% de su tamaño original
image_resized = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

# Mostrar la imagen redimensionada
cv2.imshow("Vista previa", image_resized)

# Configurar el callback del mouse para dibujar rectángulos
cv2.setMouseCallback("Vista previa", create_rectangle)

# Esperar hasta que el usuario termine de dibujar los rectángulos
while True:
    img_copy = image_resized.copy()
    for (x, y, w, h) in rectangles:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja los rectángulos
    cv2.imshow("Vista previa", img_copy)

    key = cv2.waitKey(1) & 0xFF

    # Si se presiona Ctrl + Z (tecla 26 en ASCII, con las teclas de combinación)
    if key == 26:  # Control + Z
        if rectangles:
            rectangles.pop()  # Elimina el último rectángulo
            print("Último rectángulo deshecho.")

    elif key == 27:  # Esc para salir
        break
    elif key == 13:  # Enter para guardar
        break

# Guardar las regiones detectadas en la imagen original
save_detected_regions(image, rectangles, output_dir)
print(f"Regiones detectadas: {len(rectangles)}")

cv2.destroyAllWindows()
