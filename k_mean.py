import pickle
import cv2
import numpy as np
import skfuzzy as fuzzy
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt

with open('n_51_deconvolution.pkl', 'rb') as f:
    mask = pickle.load(f)
# 1-23866-37667.png
# 5-24778-19194.png
image_b = cv2.imread('5-24778-19194.png')[:, :, ::-1]

list_pixel = []
list_location = []

for i in range(image_b.shape[0]):
    for j in range(image_b.shape[1]):
        if mask[i, j] == 1:
            list_pixel.append(image_b[i, j].astype(np.float32))
            list_location.append([i, j])

print(np.array(list_pixel).shape)
print(np.moveaxis(np.array(list_pixel), 0, -1).shape)
fpcs = []

#centroids, variance = fuzzy.cluster.cmeans(np.array(list_pixel), m=3, 3, error=0.005, maxiter=1000,init=None)
cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(np.moveaxis(np.array(list_pixel), 0, -1), 2, 2, error=0.005, maxiter=1000, init=None)
#centroids, variance = fuzzy.cluster.cmeans(np.array(list_pixel), 2)
code, distance = vq(np.array(list_pixel), cntr)
fpcs.append(fpc)

segmentation_back = np.zeros(mask.shape)
segmentation_fore = np.zeros(mask.shape)

for k in range(code.size):
    if code[k] == 0:
        segmentation_back[list_location[k][0], list_location[k][1]] = 1
    else:
        segmentation_fore[list_location[k][0], list_location[k][1]] = 1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(2, 3))
dst_1 = cv2.morphologyEx(segmentation_back, cv2.MORPH_CLOSE, kernel)
dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_OPEN, kernel)

dst_2 = cv2.morphologyEx(segmentation_fore, cv2.MORPH_CLOSE, kernel)
dst_2 = cv2.morphologyEx(dst_2, cv2.MORPH_OPEN, kernel)

plt.subplot(231)
plt.imshow(mask)
plt.title('Mask')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(segmentation_back)
plt.title('Back')
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(segmentation_fore)
plt.title('Fore')
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.imshow(image_b)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(dst_1)
plt.title('Back_Mor')
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.imshow(dst_2)
plt.title('Fore_Mor')
plt.xticks([])
plt.yticks([])

plt.show()
