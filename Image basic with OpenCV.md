# Image basic with OpenCV
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
```

## 1. Opening Image Files

**img = cv2.imread('../DATA/00-puppy.jpg')**

**plt.imshow(img_bgr)**

**img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) -- Convert Image color** 

**img_gray = cv2.imread('../DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)**

**plt.imshow(img_gray)**

**plt.imshow(img_gray,cmap='gray') -- Show the gray image**

## 2. Resize image

**(1)**
**img_rgb.shape**
**img =cv2.resize(img_rgb,(1300,275))**

**(2)**
**w_ratio = 0.5
h_ratio = 0.5**
**new_img =cv2.resize(img_rgb,(0,0),img,w_ratio,h_ratio)**

## 3. Flipping images
**new_img = cv2.flip(new_img,0) -- Along central x axis**
**new_img = cv2.flip(new_img,1) -- Along central y axis**
**new_img = cv2.flip(new_img,-1) -- Along both axis**

## 4. Saving images 
**type(new_img)**
**cv2.imwrite('my_new_picture.jpg',new_img) -- Saved BGR vision**

## 5. Large display in Notebook
**fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(new_img)**

## 6. Open file in python script
```python
# MUST BE RUN AS .py SCRIPT IN ORDER TO WORK.
# PLEASE MAKE SURE TO WATCH THE FULL VIDEO FOR THE EXPLANATION TO THIS NOTEBOOK
# TO BE CLEAR: RUNNING THIS CELL WILL KILL THE KERNEL IF YOU USE JUPYTER DIRECTLY

import cv2

img = cv2.imread('../DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
# Show the image with OpenCV
cv2.imshow('window_name',img)
# Wait for something on keyboard to be pressed to close window.
cv2.waitKey()
```