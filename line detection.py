import cv2
import numpy as np
import pre_function as pf


video = cv2.VideoCapture('실선 차선인식 test.mp4')

#라인 검출 파라미터
rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

def line_detection(src):
        lined = pf.hough_lines(src, rho, theta, threshold, min_line_len, max_line_gap)
        return lined
    
def pre_processing(src):
    frame = cv2.resize(src,(960,720))
    pre_frame = pf.pre(frame)
    edge_frame = pf.edge(pre_frame)
    masked = pf.making_mask(edge_frame,frame)
    return masked

while True :
    ret, frame = video.read()
    frame = cv2.resize(frame,(960,720))
    if not ret:
        break
    
    pre_processing_video = pre_processing(frame)
    lined_video = line_detection(pre_processing_video)
    final = cv2.addWeighted(lined_video, 1., frame, 1., 0. )

    cv2.imshow('lined_video', lined_video)
    cv2.imshow('frame', frame)
    cv2.imshow('dst', final)







    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()