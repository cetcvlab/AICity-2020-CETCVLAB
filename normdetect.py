import os 
import sys
import math
import cv2
from imageai.Detection import ObjectDetection

VIDEOPATH=open("Settings.txt",'r').readlines()[0]
WEIGHTPATH=open("Settings.txt",'r').readlines()[1]
TXTOUTPATH="Detections/"
os.mkdir(TXTOUTPATH)
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(WEIGHTPATH)
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)

for video_num in range(1,101):
    cap = cv2.VideoCapture(VIDEOPATH+str(video_num)+'.mp4')
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    framecount=0
    writelist=[]
    while(cap.isOpened()):			
        ret, frame = cap.read()
        framecount+=1
        print(framecount)
        if(not ret):
            break
        ret_img,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_type="array",input_image=frame, output_type="array", minimum_percentage_probability=10)
        for eachObject in detections:				
            if eachObject["percentage_probability"]>10.0:
                writelist.append([video_num,framecount,eachObject["box_points"],eachObject["percentage_probability"],eachObject["name"]])
    with open(TXTOUTPATH+str(video_num)+".txt","w") as outtextfile:
        for lines in writelist:
            outtextfile.write(str(lines) + "\n")
    cap.release()
