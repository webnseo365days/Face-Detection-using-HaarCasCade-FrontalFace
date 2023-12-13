import cv2 as cv
import os
import numpy as np

face_recognizer.read("face_trained.yml")

def do_Recognize(img):
    img2 = cv.resize(img, (600,600))
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+h]    
        label, confidence = face_recognizer.predict(faces_roi)
        #print("label =", label, "confidence =", confidence)
        cv.putText(img, str(people[label] ) + str(confidence) , (x,y), cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255), thickness=2)
        cv.rectangle(img, (x,y),(x+w,y+h), (255,0,0),thickness=2)
        cv.imshow('Video', img)

# Reading Videos
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if isTrue:    
        do_Recognize(frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break
 
capture.release()
cv.destroyAllWindows()
