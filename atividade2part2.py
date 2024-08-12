import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2

def apply_knn_mean_filter(image, k=3):

    N, M = image.shape
    
    output = np.zeros_like(image)

    
    for x in range(N):
        for y in range(M):
            
            neighbors = []
            for i in range(max(0, x - k), min(N, x + k + 1)):
                for j in range(max(0, y - k), min(M, y + k + 1)):
                   
                    neighbors.append(image[i, j])
            
            output[x, y] = np.mean(neighbors)

    return output


imagem1 = cv2.imread("MAR.jpg")
imagem2 = cv2.imread("lena.jpeg")

imagem1_gray = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
imagem2_gray = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)


imagem1_knn_smoothed = apply_knn_mean_filter(imagem1_gray, k=3)
imagem2_knn_smoothed = apply_knn_mean_filter(imagem2_gray, k=3)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.title("Imagem Suavizada 1 (KNN)")
plt.imshow(imagem1_knn_smoothed, cmap="gray")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.title("Imagem Suavizada 2 (KNN)")
plt.imshow(imagem2_knn_smoothed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()