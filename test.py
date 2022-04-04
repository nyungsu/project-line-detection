import cv2
import numpy as np
import line_detection as pf


video = cv2.VideoCapture('실선 차선인식 test.mp4')

while True :
    ret, frame = video.read()
    
    if not ret:
        break
    
   # ------------------------------------------------------------
    
    img = cv2.resize(frame, None, fx=1, fy=1)   # YOLO용 resize
    
    print(img.shape)
    cv2.imshow('frame', frame)
    
    

    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()