from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
from utils.image import gaussian_radius, draw_elipse_gaussian

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

h = 256
w = 256

xradius = int(gaussian_radius((120,120)))
yradius = int(gaussian_radius((60,60)))
print(xradius,yradius)

img = np.zeros((h,w), dtype=np.float32)
mask = draw_elipse_gaussian(img, (w/2, h/2), (xradius,yradius))
# mask = ((mask>0.)*255).astype(np.uint8)
cv2.imshow("test", mask)
cv2.waitKey()