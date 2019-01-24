# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:10:07 2018

@author: willi
"""

import numpy as np
import cv2
import glob
import os
import pandas as pd

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#this function returns the number of faces in an image
def detectface(arg1):
    image = cv2.imread(arg1)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(grayImage, 1.2, 2) 
    #print( type(faces))
     
    if len(faces) == 0:
        #print("No faces found")
        return 0
     
    else:
        print(faces)
        print(faces.shape)
        print( "Number of faces detected: " + str(faces.shape[0]))
        return faces.shape[0]
'''
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
            
        cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
        cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
     
        cv2.imshow('Image with faces',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''
        



a=[]
b=[]
arg = glob.glob('C:/Users/willi/Desktop/Facial Recog/Images/*')
names = os.listdir('C:/Users/willi/Desktop/Facial Recog/Images')
for i in range(len(arg)):
    a.append(detectface(arg[i]))
    b.append(os.path.getmtime(arg[i]))

print(a)
print(names)
cntlist = {'cnt':a}
jpgnames = {'jpg_name':names}
unixtime = {'lastmodifieddate':b}
#os.path.getmtime('C:/Users/willi/Desktop/facial recognition study/Images/IndependenceDayParade2.jpg')
cnts = pd.DataFrame(cntlist)
titles = pd.DataFrame(jpgnames)
timestamp = pd.DataFrame(unixtime)

finaloutput = pd.DataFrame()
finaloutput['titles'] = titles
finaloutput['cnts'] = cnts
finaloutput['time'] = timestamp
finaloutput.to_csv("test_final output.csv")
