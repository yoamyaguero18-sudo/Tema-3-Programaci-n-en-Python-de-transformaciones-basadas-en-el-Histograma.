import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1️⃣ Cargar la imagen
# Asegúrate de tener 'Lena.png' en la misma carpeta que este script
img = cv2.imread('/Users/yoamy/Escuela/Sistemas de Percepción en Robótica/Tarea Unidad 3/lena_color.tiff')

# Comprobación
if img is None:
    raise FileNotFoundError("No se encontró el archivo 'lena_color.tiff'. Asegúrate de tenerlo en la misma carpeta.")

# Convertir a escala de grises para algunas operaciones
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ============================================================
# 1. Imagen rotada
# ============================================================
(h, w) = img.shape[:2]
centro = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centro, 45, 1.0)  # Rotar 45 grados
rotada = cv2.warpAffine(img, M, (w, h))

# ============================================================
# 2. Negativo de la imagen
# ============================================================
negativo = 255 - img

# ============================================================
# 3. Imagen umbralizada (binarización)
# ============================================================
_, umbralizada = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# ============================================================
# 4. Imagen ecualizada (mejora de contraste)
# ============================================================
ecualizada = cv2.equalizeHist(gray)

# ============================================================
# 5. Eliminación de ruido (filtro Gaussiano)
# ============================================================
sin_ruido = cv2.GaussianBlur(img, (5, 5), 0)

# ============================================================
# Mostrar resultados
# ============================================================
titles = ['Original', 'Rotada', 'Negativo', 'Umbralizada', 'Ecualizada', 'Sin ruido']
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(rotada, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(negativo, cv2.COLOR_BGR2RGB),
          umbralizada,
          ecualizada,
          cv2.cvtColor(sin_ruido, cv2.COLOR_BGR2RGB)]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    if i in [3, 4]:  # Grises
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# Guardar resultados en archivos
# ============================================================
cv2.imwrite('lena_color.tiff_rotada.png', rotada)
cv2.imwrite('lena_negativa.png', negativo)
cv2.imwrite('lena_color.tiff_umbralizada.png', umbralizada)
cv2.imwrite('lena_color.tiff_ecualizada.png', ecualizada)
cv2.imwrite('lena_color.tiff_sin_ruido.png', sin_ruido)

print("✅ Imágenes procesadas y guardadas correctamente.")