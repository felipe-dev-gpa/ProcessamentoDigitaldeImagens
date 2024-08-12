import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2

def apply_mean_filter(image, kernel_size=3):
    
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    
    return cv2.filter2D(image, -1, kernel)


imagem1 = cv2.imread("MAR.jpg")
imagem2 = cv2.imread("lena.jpeg")

imagem1_gray = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
imagem2_gray = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)


imagem1_smoothed = apply_mean_filter(imagem1_gray, kernel_size=3)
imagem2_smoothed = apply_mean_filter(imagem2_gray, kernel_size=3)


plt.figure(figsize=(12, 12))


plt.subplot(2, 2, 2)
plt.title("Imagem Suavizada 1")
plt.imshow(imagem1_smoothed, cmap="gray")
plt.axis("off")


plt.subplot(2, 2, 4)
plt.title("Imagem Suavizada 2")
plt.imshow(imagem2_smoothed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()