# Image processing

```PYTHON
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

## 1. Color map -- Image's color component
Convert image's color map (BGR TO RGB)
**cv2.imread ---- color map is BGR**

```PYTHON
img = cv2.imread('../DATA/00-puppy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```

## 2. Blending and Pasting -- Combine two images 
### Blend image
```python
blended = cv2.addWeighted(src1=img1,alpha=0.7,src2=img2,beta=0.3,gamma=0)
```

### Overlaying Images of Different Sizes
We can use this quick trick to quickly overlap different sized images, by simply reassigning the larger image's values to match the smaller image.

### Blending Images of Different Sizes
1. Import
2. Importing the images again and resizing
3. Create a Region of Interest (ROI)
4. Creating a Mask
5. Convert Mask to have 3 channels
6. Grab Original FG image and place on top of Mask
7. Get ROI and blend in the mask with the ROI
8. Now add in the rest of the image


## 3. Image Thresholding -- create a limitation of pixel color number (Edge detection)
1. **Binary**
   **ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)**
   &nbsp;
2. **Binary Inverse**
   **ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)**
   &nbsp;
3. **Threshold Truncation**
   **ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)**
   &nbsp;
4. **Threshold to Zero (Inverse)**
   **ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)**
   &nbsp;
5. **Threshold to Zero (Inverse)**
   **ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)**
   &nbsp;
6. **Adaptive Threshold**
   **th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)**
   **Play around with these last 2 numbers (11,8)**

**Can blend binary threshold image and adaptive threshold image for better performance**

## 4. Blurring and Smoothing
###Blurring
1. **Gamma correction**
```python
img = load_img()
gamma = 1/4
effected_image = np.power(img, gamma)
```
&nbsp;
2. **Create a kernel or use cv function**
   **kernel = np.ones(shape=(5,5),dtype=np.float32)/25** (Low Pass Filter with a 2D Convolution)
   **dst = cv2.filter2D(img,-1,kernel)**
   or
   **blurred_img = cv2.blur(img,ksize=(5,5))**
&nbsp;
3. **Gaussian Blurring**
   **blurred_img = cv2.GaussianBlur(img,(5,5),10)**
&nbsp;
4. **Median Blurring** -- **Reduce image noise**
   **median_result = cv2.medianBlur(img,5)**
&nbsp;
5.  **BilateralFilter**--**Similar to median blur**
   **blur = cv2.bilateralFilter(img,9,75,75)**

## 5. Morphological Operators
### Erosion -- Erodes away boundaries of foreground objects

**kernel = np.ones((5,5),np.uint8)**
**erosion1 = cv2.erode(img,kernel,iterations = 1)**
**erosion5 = cv2.erode(img,kernel,iterations = 4)**

### Opening -- Opening is erosion followed by dilation. Useful in removing background noise
**opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)**

## Closing -- Useful in removing noise from foreground objects, such as black dots on top of the white text
**closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)**

## Morphological Gradient -- Difference between dilation and erosion of an image (Could be used for edge detection)
**gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)**

# 6. Gradients -- Sobel operator/Edge detection

## Sobel function (dx,dy)

**sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)** **-- 64F is the precise value, (dx,dy)= (1,0)**
**sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)** **-- 64F is the precise value, (dx,dy)= (0,1)**

**We can blend sobelx and sobely to achieve x and y line detection:**
**blended = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=0)**

## Laplacian function -- use gradient on x and y together
**laplacian = cv2.Laplacian(img,cv2.CV_64F)**

## Morphology for gradient
**kernel = np.ones((4,4),np.uint8)**

**gradient = cv2.morphologyEx(blended,cv2.MORPH_GRADIENT,kernel)**


# 7. Histograms

## Show the color distribution
**hist_values = cv2.calcHist([blue_bricks],channels=[0],mask=None,histSize=[256],ranges=[0,256]) -- Channels = [0,1,2] = Like [R,G,B] OR [B,G,R]** 

## Show three channels together
**color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,50])
    plt.ylim({0,500000})
plt.title('Histogram Forblue Horse')**

## Equalization the histograms -- To incrase the contrast
**eq_gorilla = cv2.equalizeHist(gorilla)**

**In order to deal with color img:**
**1. Convert BGR TO HSV : hsv = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2HSV)**
**2. hsv[:,:,2].min()**
**3. hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])**
**4. Convert HSV back to RGB : eq_color_gorilla = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)**