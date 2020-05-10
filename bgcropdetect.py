from imageai.Detection import ObjectDetection
import os
import cv2
import math
import numpy as np

INPATH = "./MinuteMask/"
OUTPATH = "./BGCropDetections/"

def crop_layer(x_in,nr=2,nc=4):
    columns = math.ceil(x_in.shape[1]/nc)
    rows = math.ceil(x_in.shape[0]/nr)
    img_new = cv2.resize(x_in, (int(columns*nc), int(rows*nr)))
    grid = []
    for i in range(nr):
        for j in range(nc):
            grid.append(img_new[i*rows:(i+1)*rows,j*columns:(j+1)*columns,:])
    return grid

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "./yolo.h5"))
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)

for i in range(1,101):
    Len = len(os.listdir(os.path.join(INPATH,str(i))))
    if(not os.path.exists(OUTPATH+str(i))): 
            os.mkdir(OUTPATH+str(i))
    texfile=open(OUTPATH+str(i)+"/out.txt","w")
    for q in range(Len):        
        img=cv2.imread(INPATH+str(i)+'/'+str(q+1)+'.png')
        imlist=crop_layer(img)
        for index,crop in enumerate(imlist):        
            ret_img,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_type="array",input_image=crop, output_type="array", minimum_percentage_probability=10)
            for eachObject in detections:
                box=eachObject["box_points"]
                if eachObject["percentage_probability"]>10.0:
                    texfile.write(str(i)+"," +str(q)+","+str(index)+","+str(eachObject["percentage_probability"])+","+str(eachObject["box_points"])+","+str(eachObject["name"] )+ "\n")



