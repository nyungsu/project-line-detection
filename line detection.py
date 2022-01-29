import cv2
import sys
import numpy as np


def pre(src):
    src_cpy = src.copy()    
    gray = cv2.cvtColor(src_cpy,cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(0, 0), 3)
    return blur_gray

def edge(src):
    src_cpy = src.copy()
    return cv2.Canny(src, 10, 30)

def draw_lines(src, lines, color=[0,0,255],thickness=5):
    for line in lines :
        for x1,y1,x2,y2 in line :
            cv2.line(src, (x1, y1,), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, 
    threshold, 
    minLineLength=min_line_len, 
    maxLineGap=max_line_gap)
    
    

    line_img = np.zeros((img.shape[0],img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


video = cv2.VideoCapture('실선 차선인식 test.mp4')

while True :
    ret, frame = video.read()

    if not ret:
        break
    
    frame = cv2.resize(frame,(960,720))
    pre_frame = pre(frame)
    edge_frame = edge(pre_frame)
    
    #마스크 도형 만들기
    black = np.zeros_like(frame)
    pts = np.array([[550,312], [400,312], 
                [60,580], [960,580]])
    mask = cv2.fillPoly(black, [pts], (255,255,255), cv2.LINE_AA)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    masked = cv2.bitwise_and(edge_frame, mask)

    #라인 검출 파라미터
    rho = 2
    theta = np.pi/180
    threshold = 90
    min_line_len = 120
    max_line_gap = 150

    lined = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
    
    dst = cv2.addWeighted(lined, 1., frame, 1., 0. )

   
    cv2.imshow('frame', frame)
    cv2.imshow('dst', dst)







    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()