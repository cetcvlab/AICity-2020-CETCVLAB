from imageai.Detection import ObjectDetection
import os
import cv2

inpath = "./MinuteMask/"
outpath = "./BGDetections/"
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "./yolo.h5"))
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)

for i in range(1,101):
    Len = len(os.listdir(os.path.join(inpath,str(i))))
    if(not os.path.exists(outpath+str(i))): 
        os.mkdir(outpath+str(i))
    texfile=open(outpath+str(i)+"/out.txt","w")
    for q in range(Len):            
        detections = detector.detectCustomObjectsFromImage( custom_objects=custom,input_image=os.path.join(execution_path , inpath+str(i)+"/"+str(q+1)+'.png'), output_image_path=os.path.join(execution_path , outpath+str(i)+"/"+str(q+1)+'.png'), minimum_percentage_probability=10)
        for eachObject in detections:					
            if eachObject["percentage_probability"]>10.0:
                texfile.write(str(i)+"," +str(q)+", "+str(eachObject["percentage_probability"])+","+str(eachObject["box_points"])+","+str(eachObject["name"] )+ "\n")
