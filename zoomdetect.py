import os 
import sys
import math
import cv2
from imageai.Detection import ObjectDetection

TXTOUTPATH= "CropDetections/"
VIDEOPATH="AIC20_track4/test-data/"
WEIGHTPATH="yolo.h5"

os.mkdir(TXTOUTPATH)
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(WEIGHTPATH)
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)

Zoomlist=[3, 9, 13, 17, 20, 24, 25, 28, 29, 32, 34, 41, 45, 54, 66, 75, 81, 82, 85, 90, 92, 93, 95, 96]

def crop_layer(x_in,nr,nc):
    columns = math.ceil(x_in.shape[1]/nc)
    rows = math.ceil(x_in.shape[0]/nr)
    img_new = cv2.resize(x_in, (int(columns*nc), int(rows*nr)))
    grid = []
    for i in range(nr):
        for j in range(nc):
            grid.append(img_new[i*rows:(i+1)*rows,j*columns:(j+1)*columns,:])
    return grid

for video_num in Zoomlist:
    cap = cv2.VideoCapture(VIDEOPATH+str(video_num)+'.mp4')
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    framecount=0
    writelist=[]
    while(cap.isOpened()):			
        ret, frame = cap.read()
        framecount+=1
        cropnum=-1
        imlist=crop_layer(frame,2,4)
        if(not ret):
            break
        for crops in imlist:
            ret_img,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_type="array",input_image=crops, output_type="array", minimum_percentage_probability=10)
            cropnum+=1
            for eachObject in detections:				
                if eachObject["percentage_probability"]>10.0:
                        writelist.append([video_num,framecount,cropnum,eachObject["box_points"],eachObject["percentage_probability"],eachObject["name"]])
    with open(TXTOUTPATH+str(video_num)+".txt","w") as outtextfile:
        for lines in writelist:
            outtextfile.write(str(lines) + "\n")
    cap.release()
