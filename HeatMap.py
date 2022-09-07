# Some sort of summarizing comment needed here to introduce users and/or new collaborators

import cv2
import urllib.request
import json
import pandas as pd
import datetime
import time
import numpy as np
import subprocess
import pytesseract
import matplotlib.pyplot as plt
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# FUNCTION takes the date and time from the file_name and convert it to the unix time format for the event current time synchronyzation 
file_name = input("Enter the File Name: ")
video_file = open(file_name, "r")
#file_name ='2021-12-06-20.43.50 maxCurrent 12.627 uA fromVideoStarting 20211206-20.40.00..mp4'       #To Decrease redundency to type file-name
video_file = open(file_name,"r")

thresholds = 252
# import the video file form the system and look for the corresponding events
cap = cv2.VideoCapture(file_name)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
print("frame dimentions: ", width, height )
object_detector = cv2.createBackgroundSubtractorMOG2()   #Extract the background
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('file_data.mp4',fourcc,20,(1920,1080), True)
fps = cap.get(cv2.CAP_PROP_FPS) 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
duration = frame_count/fps
# output video file is produced in .mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('file_data2.mp4',fourcc,7,(1920,1080))
# print the total frame count in the video file
print("Number of Frame: "+str(frame_count))
# print the total duration of a video file
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration%60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
count =0
frame_count = 0
date_data = []
while True:
    ret, frame=cap.read()
    if ret == True:
        #frame[frame<=thresholds]=0
        mask = object_detector.apply(frame)
        # mask the backgroubd to the color ratio of 255 for complete black color 
        _, mask  = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = cv2.bitwise_and(frame,frame,mask=mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # detect flashes above 1000pixel
            if area>1500:
                blur = cv2.GaussianBlur(mask,(13,13), 11)
                heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap_img, (heatmap_img.shape[1], heatmap_img.shape[0]))
                #cv2.imshow("heat",heatmap)
                cv2.imshow("heatmap", heatmap_img)
                cv2.imwrite("file%d.jpg"%count, heatmap_img)
                out.write(heatmap_img)
                count +=1
    else:
        break
    if cv2.waitKey(8) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
