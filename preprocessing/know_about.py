import cv2
import os
import matplotlib.pyplot as plt

image = cv2.imread(r"D:\DeepCrack\dataset\train_img\7Q3A9060-14.jpg")
mask = cv2.imread(r"D:\DeepCrack\dataset\train_lab\7Q3A9060-14.png", cv2.IMREAD_GRAYSCALE)

print("Image loaded:", image is not None)
print("Mask loaded:", mask is not None)

image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.show()
