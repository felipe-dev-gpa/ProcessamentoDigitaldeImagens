import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os

def apply_prewitt(image):
    #Definindo os kernels de Prewitt
    kernel_x = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])

    kernel_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])


    gx = cv2.filter2D(image, -1, kernel_x)
    gy = cv2.filter2D(image, -1, kernel_y)


    edges = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))

    return edges


imagem1 = cv2.imread("MAR.jpg")
imagem2 = cv2.imread("lena.jpeg")

if imagem1 is None:
    print("Erro ao carregar MAR.jpg")
else:
    imagem1_gray = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)

if imagem2 is None:
    print("Erro ao carregar lena.jpeg")
else:
    imagem2_gray = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)


imagem1_prewitt = apply_prewitt(imagem1_gray) if imagem1_gray is not None else None
imagem2_prewitt = apply_prewitt(imagem2_gray) if imagem2_gray is not None else None


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.title("Detector de Bordas de Prewitt - Imagem 1")
plt.imshow(imagem1_prewitt, cmap="gray")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.title("Detector de Bordas de Prewitt - Imagem 2")
plt.imshow(imagem2_prewitt, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
