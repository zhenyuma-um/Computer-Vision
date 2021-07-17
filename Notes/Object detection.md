# Object detection
## 1. Templete matching -- Using a subset of a large image to match in a large image

**Lib**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
**Read two images**
```python
full = cv2.imread('DATA/sammy.jpg')
full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)
face = cv2.imread('DATA/sammy_face.jpg')
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
```

**All the 6 methods for comparison in a list
Note how we are using strings, later on we'll use the eval() function to convert to function**

**methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']**


**Use templete matching and draw a rectangle:**
```python
for m in methods:
    
    # CREATE A COPY
    full_copy = full.copy()
    
    method = eval(m)
    
    # TEMPLATE MATCHING
    res = cv2.matchTemplate(full_copy,face,method)
    
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    height,width,channels = face.shape
    
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(full_copy,top_left,bottom_right,(255,0,0),10)
    
    # PLOT AND SHOW THE IMAGES
    
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heatmap of templete matching')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detection of templete')
    # Title with the method used
    plt.suptitle(m)
    
    plt.show()
    
    print('\n')
    print('\n')
```

**Single method**
```python
my_method = eval('cv2.TM_CCOEFF') # Make my_method = the function
res = cv2.matchTemplate(full,face,my_method)
plt.imshow(res)
```


## 2. Corner detection 
### Harris corner -- detect by major direction change
```python
flat_chess = cv2.imread('DATA/flat_chessboard.png')
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)
flat_chess[dst>0.01*dst.max()] = [255,0,0] #RGB mark the corner
plt.imshow(flat_chess)
```
**Or**
```python
real_chess = cv2.imread('DATA/real_chessboard.jpg')
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)
real_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(real_chess)
```

### Shi-Tomasi detection -- easier method
```python
flat_chess = cv2.imread('DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

real_chess = cv2.imread('DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_flat_chess,64,0.01,10) # 64 is the detect number
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(flat_chess,(x,y),3,(255,0,0),-1)

plt.imshow(flat_chess)

corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(real_chess,(x,y),3,(255,0,0),-1)

plt.imshow(real_chess)
```

## 3. Edge detection -- Canny function
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('DATA/sammy_face.jpg')
edges = cv2.Canny(image = img, threshold1=0, threshold2=255)
```

**We can decide the threshold by med value**
```python
med_val = np.median(img)

# Low threshold to either 0 or 70% of the median value whichever is greater
lower = int(max(0,0.7*med_val))
# Upper threshold to either 130% of the median or the max 255, whichever is smaller
upper = int(min(255,1.3*med_val)) 

edges = cv2.Canny(image = img, threshold1=lower, threshold2=upper)
```
**To get a good detection result, we can blur image first**
```python
blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image = blurred_img, threshold1=lower, threshold2=upper)
```

## 4. Grid detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


flat_chess = cv2.imread('DATA/flat_chessboard.png')
found,corners = cv2.findChessboardCorners(flat_chess,(7,7))
cv2.drawChessboardCorners(flat_chess,(7,7),corners,found)
```
**Used for camera calibration**
```python
dots = cv2.imread('DATA/dot_grid.png')
found,corners = cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)
cv2.drawChessboardCorners(dots,(7,7),corners,found)
```

## 5. Contour detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('DATA/internal_external.png',0)
image,contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
```

**Detect external contour:**
```python
external_contours = np.zeros(image.shape)

for i in range(len(contours)):
    
    # EXTERNAL
    if hierarchy[0][i][3] == -1: #This number depends on hierarchy
        
        cv2.drawContours(external_contours,contours,i,255,-1)

plt.imshow(external_contours,cmap='gray')
```

**Detect internal contour:**
```python
internal_contours = np.zeros(image.shape)

for i in range(len(contours)):
    
    # Internal
    if hierarchy[0][i][3] != -1: # Not equal to -1 means internal contours
        
        cv2.drawContours(internal_contours,contours,i,255,-1)
    
plt.imshow(internal_contours,cmap='gray')
```

## 6. Feature matching - Use to one small image to find the matching feature on a large image
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

reeses = cv2.imread('../DATA/reeses_puffs.png',0)  
cereals = cv2.imread('../DATA/many_cereals.jpg',0) 
```

### Brute Force Detection with ORB Descriptors
```python
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(reeses,None)
kp2, des2 = orb.detectAndCompute(cereals,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 25 matches.
reeses_matches = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:25],None,flags=2)

display(reeses_matches)
```


### Brute-Force Matching with SIFT Descriptors and Ratio Test
```python
# Create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

# cv2.drawMatchesKnn expects list of lists as matches.
sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
```

### FLANN based Matcher
```python
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []

# ratio test
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.7*match2.distance:
        
        good.append([match1])


flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=0)

display(flann_matches)
```
### FLANN based Matcher + Mask
```python
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.7*match2.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)

display(flann_matches)
```