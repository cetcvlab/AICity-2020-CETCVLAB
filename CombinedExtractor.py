import os 
import sys
import ast
import matplotlib.pyplot as plt
import cv2
from statistics import variance

extractor=""

RESPATH="Result.txt"

if len(sys.argv) > 2:
    print('You have specified too many arguments')
    sys.exit()

if len(sys.argv) < 2:
    extractor = "both"    
    if os.path.exists(RESPATH):
        os.remove(RESPATH)    
else :
    extractor = sys.argv[1]

RESULTFILE=open(RESPATH,"a+")

YTHRESH=100
SIZETHRESH=200
IOUTHRESH=0.3
VARTHRESH=100

def erodeanddilate(score):
        erkernelsize=10
        dilkernelsize=20
        kernel=[0]*erkernelsize
        for i in range(0,len(score)-len(kernel),len(kernel)):
                if(sum(score[i:i+len(kernel)])<(len(kernel)-1)):
                        score[i:i+len(kernel)]=kernel
        kernel=[1]*dilkernelsize
        for i in range(0,len(score)-len(kernel),len(kernel)):
                if(sum(score[i:i+len(kernel)])>0):
                        score[i:i+len(kernel)]=kernel
        return score

def zoomerodeanddilate(score):
        erkernelsize=100
        dilkernelsize=200
        kernel=[0]*erkernelsize
        for i in range(0,len(score)-len(kernel),len(kernel)):
                if(sum(score[i:i+len(kernel)])<(len(kernel)/2)):
                        score[i:i+len(kernel)]=kernel
        kernel=[1]*dilkernelsize
        for i in range(0,len(score)-len(kernel),len(kernel)):
                if(sum(score[i:i+len(kernel)])>0):
                        score[i:i+len(kernel)]=kernel
        return score

def bb_intersection_over_union(boxA, boxB):
    if(boxA[1]<YTHRESH or boxB[1]<YTHRESH or (boxA[2] - boxA[0])>SIZETHRESH or (boxA[3] - boxA[1])>SIZETHRESH or (boxB[2] - boxB[0])>SIZETHRESH or (boxB[3] - boxB[1])>SIZETHRESH):
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if(iou>IOUTHRESH):
        return iou
    else:
        return 0

def zoombb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if(iou>IOUTHRESH):
        return iou
    else:
        return 0

def filterbgbox(bgbx):
    finalboxes=[]
    for minute in bgbx:
        boxlist=bgbx[minute]
        centroid=[[],[]]
        for boxs in boxlist:
            if(boxs[1]<YTHRESH or (boxs[2] - boxs[0])>SIZETHRESH or (boxs[3] - boxs[1])>SIZETHRESH ):
                boxlist.remove(boxs)
            centroid[0].append((boxs[0]+boxs[2])/2)
            centroid[1].append((boxs[1]+boxs[1])/2)
        if(len(centroid[0])>5):
            var=(variance(centroid[0]),variance(centroid[1]))
        else:
            var=(0,0)
        if(var[0]<VARTHRESH and var[1]<VARTHRESH):
            for box in boxlist:
                finalboxes.append(box)
    return finalboxes

def calcNormScore():
    outfile = open("framescore.txt", 'w')
    anofile = open("anomaly.txt", 'w')
    dark = open("dark.txt", 'r').readlines()
    freeze = open("freeze.txt", 'r').readlines()
    darkfreeze= list(map(int,dark))

    for video in freeze:
        darkfreeze.append(int(video.split(" ")[0]))
    anomalyCanidate=[]

    for i in range(1,101):
        print(i)
        if(os.stat('BGDetections/'+str(i)+'/out.txt').st_size==0 or i in darkfreeze):
            anomalyCanidate.append(i)
            continue
        detfile = open('Detections/'+str(i)+'.txt', 'r')
        anomalyCanidate.append(i)
        bgdetfile = open('BGDetections/'+str(i)+'/out.txt', 'r')
        bglines = bgdetfile.readlines()
        bgbox = {}
        framescore=[0]*30000
        prevframe=0
        for lines in bglines:
            if(int(lines.split(",")[1])==0):
               continue
            if(int(lines.split(",")[1]) != prevframe):
                bgbox[int(lines.split(",")[1])]=[]
            bgbox[int(lines.split(",")[1])].append(list(map(int,lines[lines.index('[')+1:lines.index(']')].split(","))))
            prevframe=int(lines.split(",")[1])
        bgbox=filterbgbox(bgbox)
        while True:
            line = detfile.readline()
            if not line: 
                break
            line=ast.literal_eval(line)
            frame=line[1]
            box=line[2]
            for bbox in bgbox:
                framescore[frame]+=bb_intersection_over_union(box,bbox)
        outfile.write(str(i) +"\t"+str(framescore)+"\n")
        detfile.close()
        bgdetfile.close()
    for anomaly in anomalyCanidate:
        anofile.write(str(anomaly)+"\n")
    anofile.close()

def processNormScore():
    TXTINPATH="framescore.txt"
    anomalies=list(map(int,open("anomaly.txt",'r').readlines()))
    zoomcheck=open("zoomcheck.txt","w")

    with open(TXTINPATH, 'r') as read_file:
            readlines= read_file.readlines()
            for k in range(len(readlines)):
                score=readlines[k].split('\t')
                framescore=ast.literal_eval(score[1])
                if(max(framescore)==0 or (int(score[0]) not in anomalies) or variance(framescore)<0.1):
                    if(max(framescore)==0 and (int(score[0]) in anomalies)):
                        zoomcheck.write(str(score[0])+"\n")
                    continue
                maxscr=max(framescore)
                framescore= [x / maxscr for x in framescore]
                for i in range(len(framescore)):
                        if(framescore[i]<0.09):
                                framescore[i]=0
                        else:
                                framescore[i]=1
                framescore=erodeanddilate(framescore)
                if(max(framescore)==0 or sum(framescore)<180):
                    if(max(framescore)==0):
                        zoomcheck.write(str(score[0])+"\n")
                    continue
                firstframe=0
                for i in range(len(framescore)):
                        if(framescore[i]==1):
                                firstframe=i
                                break
                if(firstframe<1350 or firstframe>16200):
                    continue
                RESULTFILE.write(str(score[0])+" "+str(round(firstframe/30.0,4))+" 1\n")
    zoomcheck.close()

def calcZoomScore():
    outfile = open("cropframescore.txt", 'w')
    anolist = list(map(int,open("zoomcheck.txt", 'r').readlines()))

    for i in anolist:
        detfile = open('CropDetections/'+str(i)+'.txt', 'r')
        bgdetfile = open('BGCropDetections/'+str(i)+'/out.txt', 'r')
        bglines = bgdetfile.readlines()
        bgbox = {}
        framescore=[0]*30000
        prevframe=0
        for lines in bglines:
            if(int(lines.split(",")[1])==0):
               continue
            if(int(lines.split(",")[1]) != prevframe):
                bgbox[int(lines.split(",")[1])]=[]
            bgbox[int(lines.split(",")[1])].append([list(map(int,lines[lines.index('[')+1:lines.index(']')].split(","))),int(lines.split(",")[2][1])])
            prevframe=int(lines.split(",")[1])
        while True:
            line = detfile.readline()
            if not line: 
                break
            line=ast.literal_eval(line)
            frame=line[1]
            crop=line[2]
            box=line[3]
            for minut in bgbox:
                for bosk in bgbox[minut]:
                    if(crop==bosk[1]):
                        framescore[frame]+=zoombb_intersection_over_union(box,bosk[0])
        outfile.write(str(i) +"\t"+str(framescore)+"\n")
        detfile.close()
        bgdetfile.close()
    outfile.close()

def processZoomScore():
    TXTINPATH="cropframescore.txt"
    anomalies=list(map(int,open("zoomcheck.txt",'r').readlines()))

    with open(TXTINPATH, 'r') as read_file:
            readlines= read_file.readlines()
            for k in range(len(readlines)):
                score=readlines[k].split('\t')
                print(score[0])
                framescore=ast.literal_eval(score[1])
                if(max(framescore)==0 or (int(score[0]) not in anomalies)):
                    continue
                maxscr=max(framescore)
                framescore= [x / maxscr for x in framescore]
                for i in range(len(framescore)):
                        if(framescore[i]<0.3):
                                framescore[i]=0
                        else:
                                framescore[i]=1
                framescore=zoomerodeanddilate(framescore)
                if(max(framescore)==0 or sum(framescore)<240):
                    continue
                firstframe=0
                for i in range(len(framescore)):
                        if(framescore[i]==1):
                                firstframe=i
                                break
                if(firstframe<1350 or firstframe>16200 and not(variance(framescore)<0.01)):
                    continue
                RESULTFILE.write(str(score[0])+" "+str(round(firstframe/30.0,4))+" 1\n")
    RESULTFILE.close()

if __name__ == "__main__":
    if(extractor=="both"):
        calcNormScore()
        processNormScore()
        calcZoomScore()
        processZoomScore()
    elif(extractor=="normal"):
        calcNormScore()
        processNormScore()
    elif(extractor=="zoom"):
        calcZoomScore()
        processZoomScore()
