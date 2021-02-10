from PIL import Image
import numpy as np
import cv2
from scipy import ndimage

mask = Image.open('datasets/mask/mask.jpg')
content = Image.open('datasets/mask/naive.jpg')
# mask = mask.convert('L').save('datasets/mask/mask2.jpg')
# mask.show()
mask = np.array(mask) / 255
content = np.array(content)

out = (mask * content).astype(np.uint8)
out = Image.fromarray(out)
# out.save('datasets/mask/content.jpg')
# qwe

mask = 255 - mask
mask = ndimage.grey_erosion(mask, size=(30, 30))
# print(mask)
# print(mask.shape)
# Image.fromarray(mask).show()

print(mask.shape, mask.dtype)

dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
print(dist_transform)
out = Image.fromarray(dist_transform)
out.show()
