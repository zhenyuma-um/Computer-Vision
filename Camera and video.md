# Camera and video operation
## 1. Connect to camera -- capture and write 

```python
import cv2


cap = cv2.VideoCapture(0) # 0 is default camera

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1080.0
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# WINDOWS -- *'DIVX'
# MACOS or LINUX *'XVID'

writer = cv2.VideoWriter('mysupervideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'),20,(width,height))


while True:
    
    ret,frame = cap.read()
    
    # OPERATIONS (DRAWING1)
    
    writer.write(frame)
    
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
writer.release
cv2.destroyAllWindows()
```

## 2. Use video files
```python
import cv2
import time

cap = cv2.VideoCapture('DATA/hand_move.mp4')

if cap.isOpened() == False:
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED')
    
while cap.isOpened():
    
    ret,frame = cap.read()
    
    if ret == True:
        
        # WRITER 20 FPS
        time.sleep(1/20)
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
                                         
cap.release()
cv2.destroyAllWindows()
```

## 3. Draw on videos
```python
import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TOP LEFT CORNER
x = width // 2 # return int number
y = height // 2

# width and height of rectangle
w = width // 4
h = height // 4

# Bottom right x + w, y + h


while True:
    
    ret, frame = cap.read()
    
    cv2.rectangle(frame,(x,y),(x+w,y+h),color = (0,0,255), thickness = 4)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```