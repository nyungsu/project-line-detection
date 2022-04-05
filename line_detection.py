import cv2
import sys
import numpy as np

def pre(src):
    '''
    영상에서 기본적인 전처리를 진행합니다.
    영상의 frame을 입력값으로 받고,
    흑백 이미지로 변환 후,
    가우시안 블러를 통해 노이즈를 줄인 영상을 출력합니다.
    '''
    src_cpy = src.copy()    
    gray = cv2.cvtColor(src_cpy,cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(0, 0), 3)
    return blur_gray

def edge(src):
    '''
    전처리 된 영상에서 edge를 추출합니다.
    영상을 입력으로 받고
    Canny 함수를 통한 edge가 추출 된 영상을 출력합니다.
    '''
    src_cpy = src.copy()
    return cv2.Canny(src, 10, 30)

def draw_lines(src, lines, color=[0,0,255],thickness=5):
    '''
    허프라인 함수에 이용될 빨간 줄을 그어줄 함수입니다.
    '''
    for line in lines :
        for x1,y1,x2,y2 in line :
            cv2.line(src, (x1, y1,), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    '''
    cv2.HoughLines보다 높은 정확도를 위해 HoughLinesP를 사용했습니다.
    입력 이미지 크기와 같은 크기의 검정 영상에
    draw_lines함수와 HoughLinesP 출력값을 이용하여
    빨간 선을 긋습니다.
    선이 그어진 영상을 출력합니다.
    '''    
    lines = cv2.HoughLinesP(img, rho, theta, 
                            threshold, 
                            minLineLength=min_line_len, 
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0],img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def making_mask(dst,src):
    '''
    연산을 줄이기 위하여 차선 검출이 필요한 영역을 RoI로 지정하였습니다.
    사다리꼴로 정의하였고 좌표는 실험적 데이터에 기반하여 찍었습니다.
    '''
    black = np.zeros_like(src)
    pts = np.array([[550,312], [400,312], 
                [60,580], [960,580]])
    mask = cv2.fillPoly(black, [pts], (255,255,255), cv2.LINE_AA)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(dst, mask)
    return masked
    
def pre_processing(src):
    frame = cv2.resize(src,(960,720))
    pre_frame = pre(frame)
    edge_frame = edge(pre_frame)
    masked = making_mask(edge_frame,frame)
    return masked

def drawing_line(src, rho, theta, threshold, min_line_len, max_line_gap):
        lined = hough_lines(src, rho, theta, threshold, min_line_len, max_line_gap)
        return lined