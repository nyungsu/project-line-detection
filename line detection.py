import cv2
import sys
import numpy as np
import pre_function as pf





video = cv2.VideoCapture('실선 차선인식 test.mp4')

#라인 검출 파라미터
rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

def line_detectio(src):
        lined = pf.hough_lines(src, rho, theta, threshold, min_line_len, max_line_gap)
        return lined

while True :
    ret, frame = video.read()
    frame = cv2.resize(frame,(960,720))
    if not ret:
        break
    
    pre_processing_video = pf.pre_processing(frame)
    lined_video = line_detectio(pre_processing_video)
    final = cv2.addWeighted(lined_video, 1., frame, 1., 0. )

   
    cv2.imshow('frame', frame)
    cv2.imshow('dst', final)







    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()