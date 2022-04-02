import cv2
import numpy as np
import pre_function as pf
# import object_detection as od


video = cv2.VideoCapture('실선 차선인식 test.mp4')

#라인 검출 파라미터
rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

def drawing_line(src):
        lined = pf.hough_lines(src, rho, theta, threshold, min_line_len, max_line_gap)
        return lined
    
def pre_processing(src):
    frame = cv2.resize(src,(960,720))
    pre_frame = pf.pre(frame)
    edge_frame = pf.edge(pre_frame)
    masked = pf.making_mask(edge_frame,frame)
    return masked

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


while True :
    ret, frame = video.read()
    frame = cv2.resize(frame,(960,720))     # line drawing용 resize
    if not ret:
        break
    
   # ------------------------------------------------------------
    
    img = cv2.resize(frame, None, fx=1, fy=1)   # YOLO용 resize
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        label = str(classes[i])
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    
# ------------------------------------------------------------
    pre_processing_video = pre_processing(frame)
    lined_video = drawing_line(pre_processing_video)
    final = cv2.addWeighted(lined_video, 1., img, 1., 0. )
    
    
    # cv2.imshow('lined_video', lined_video)
    # cv2.imshow('od', img)
    cv2.imshow('frame', frame)
    cv2.imshow('dst', final)
    







    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()