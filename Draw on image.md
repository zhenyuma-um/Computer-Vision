# Draw on image
```pyton
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
```
## 1. Shapes
### Rectangle
**pt1 = top left
pt2 = bottom right
cv2.rectangle(blank_img,pt1=(384,0),pt2=(510,128),color=(0,255,0),thickness=5)**

### Circles
**cv2.circle(img=blank_img, center=(100,100), radius=50, color=(255,0,0), thickness=5)**
**cv2.circle(img=blank_img, center=(400,400), radius=50, color=(255,0,0), thickness=-1) -- Filled circle**

### Lines
**cv2.line(blank_img,pt1=(0,0),pt2=(511,511),color=(102, 255, 255),thickness=5)**

### Text
**font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img,text='Hello',org=(10,500), fontFace=font,fontScale= 4,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)**

### Polygons
**vertices = np.array([[100,300],[200,200],[400,300],[200,400]],np.int32)**
**pts = vertices.reshape((-1,1,2))**
**cv2.polylines(blank_img,[pts],isClosed=True,color=(255,0,0),thickness=5)**
