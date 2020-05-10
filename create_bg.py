import cv2
import os
import numpy as np
import argparse
import uuid
import scipy.spatial
import matplotlib.pyplot as plt

ROADMASKDIR = "./RoadMask/"
MINUTEMASKDIR = "./MinuteMask/"
INPUTVIDEOPATH = os.environ['AICITYVIDEOPATH'] + "/test-data/"
darktexfile=open("dark.txt","w")
darkthreshold=290000

print("Using Input Video Path : "+INPUTVIDEOPATH)

def unsharp_mask(image, kernel_size=(7, 7), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def apply_filter(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    ret, frame = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY)
    return frame

def mkdir_ifndef(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def create_bg(vidnum):    
    mkdir_ifndef(ROADMASKDIR)
    mkdir_ifndef(MINUTEMASKDIR)
    cap = cv2.VideoCapture(INPUTVIDEOPATH+str(vidnum)+".mp4")
    vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    weight=255.0/length
    vroi = 255 * np.ones((vw, vh), dtype=np.uint8)
    vroi2 = 255 * np.ones((vw, vh), dtype=np.uint8)
    bs = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    bs.setBackgroundRatio(0.6)
    bs.setHistory(256)
    bs.setNMixtures(4)
    bs.setVarInit(15)
    bs.setVarThreshold(25)
    cmpx_reduction_frames = 256
    learn_rate=0.007
    cmpx_reduction_factor = 1 - np.exp(256 * np.log(0.995))
    masksum = np.zeros((vw, vh), np.float32)        
    (rAvg, gAvg, bAvg) = (None, None, None)
    maskcount=0
    total=0
    while True:
        ret, frame = cap.read()
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret:
            break
        if frame_num == bs.getHistory():
            learn_rate = 0.005
            bs.setComplexityReductionThreshold(cmpx_reduction_factor)
        frame = cv2.bitwise_and(frame, frame, mask=vroi)
        fg_img = bs.apply(frame, learningRate=learn_rate)
        bg_img = bs.getBackgroundImage()
        ret, fg_img = cv2.threshold(fg_img, 192, 255, cv2.THRESH_BINARY)
        fg_mask = apply_filter(fg_img)
        fg_mask2 = fg_mask.copy()
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=vroi2)
        sharpened_image = unsharp_mask(bg_img)
        kernel = np.ones((5,5), np.uint8) 
        img_erosion = cv2.erode(fg_mask, kernel, iterations=3) 
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
        opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel)
        masksum=masksum+(opening*weight)
        (B, G, R) = cv2.split(sharpened_image.astype("float"))
        if rAvg is None:
            rAvg = R
            bAvg = B
            gAvg = G
        else:
            rAvg = ((total * rAvg) + (1 * R)) / (total + 1.0)
            gAvg = ((total * gAvg) + (1 * G)) / (total + 1.0)
            bAvg = ((total * bAvg) + (1 * B)) / (total + 1.0)
        total+=1
        if(frame_num%(60*30)==0):
            maskcount+=1
            mkdir_ifndef(MINUTEMASKDIR+str(vidnum))
            total=0
            avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")
            cv2.imwrite(MINUTEMASKDIR+str(vidnum)+"/"+str(maskcount)+".png",avg)
            (rAvg, gAvg, bAvg) = (None, None, None)
            if(maskcount==1):
                img=plt.imread(MINUTEMASKDIR+str(vidnum)+"/"+str(maskcount)+".png")
                intensity = img.sum(axis=2)
                pixelsum=0
                for row in intensity:
                    pixelsum+=sum(row)
                if(pixelsum < darkthreshold):
                    darktexfile.write(str(vidnum)+"\n")
    masksum=apply_filter(masksum)          
    cv2.imwrite(ROADMASKDIR+str(vidnum)+".png",masksum)
    cap.release()

def find_freeze():
    out = open("freeze.txt",'w')
    for i in range(1,101):
        count = 1
        videoPath = INPUTVIDEOPATH + "%d.mp4"%(i)
        cap = cv2.VideoCapture(videoPath)
        ret, frame2 = cap.read()
        start = -1
        consec = 0
        while(cap.isOpened()):
            frame1 = frame2
            ret, frame2 = cap.read()
            if not ret:
                break
            count +=1
            difference = cv2.subtract(frame1, frame2)
            b, g, r = cv2.split(difference)
            if cv2.countNonZero(b) <= 3000 and cv2.countNonZero(g) <= 3000 and cv2.countNonZero(r) <= 3000:
                if(start == -1):
                    start = count - 1
                    consec = 0
            elif(start != -1):
                consec += 1
                if(consec > 10):
                    if(count - start - consec > 120):
                        out.write("%d %d %d\n"%(i, start, count-1-consec))
                    start = -1
                    consec = 0
        if(start != - 1 and start != count -1):
            start = - 1
    out.close()

if __name__ == "__main__":
    for i in range(1,101):
        create_bg(i)
    find_freeze()
