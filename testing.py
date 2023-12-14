#This line imports the OpenCV library and gives it the alias cv. The cv2 module provides computer vision functionalities, and in this case, it is used for image and video processing.
import cv2 as cv
# This line imports the os module, which provides a way to interact with the operating system, including functions for working with file systems. However, in this code, the os module is not explicitly used.
import os
# This line imports the NumPy library and gives it the alias np. NumPy is a powerful library for numerical operations in Python.
import numpy as np

#This line reads a pre-trained face recognizer model from the YAML file "face_trained.yml". The face recognizer is assumed to have been trained previously and saved to this file using the save method.
face_recognizer.read("face_trained.yml")

#This line defines a function named do_Recognize that takes an image (img) as input. This function will be used to perform face recognition on the input image.
def do_Recognize(img):
    #This line resizes the input image (img) to a fixed size of 600x600 pixels using the cv.resize function.
    img2 = cv.resize(img, (600,600))
    # It converts the resized image to grayscale using the cv.cvtColor function. Grayscale images are often used in face recognition tasks.
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    #This line creates a Haar Cascade Classifier for detecting frontal faces. The XML file containing the pre-trained classifier is loaded using cv.CascadeClassifier.
    haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    # It detects faces in the grayscale image using the Haar Cascade classifier (haar_cascade). Detected faces are represented by rectangles, and their coordinates are stored in the faces_rect variable.
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    # This line initiates a loop that iterates over the rectangles representing the detected faces.
    for (x,y,w,h) in faces_rect:
        # It extracts the region of interest (ROI) for each detected face from the grayscale image.
        faces_roi = gray[y:y+h, x:x+h]  
        # The face recognizer (face_recognizer) predicts the label and confidence level for the current face ROI using the predict method.
        label, confidence = face_recognizer.predict(faces_roi)
        
        # It puts text on the original image (img) indicating the recognized person's name (people[label]) and the confidence level.
        cv.putText(img, str(people[label] ) + str(confidence) , (x,y), cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255), thickness=2)
        # It draws a rectangle around the detected face on the original image to highlight the region of interest.
        cv.rectangle(img, (x,y),(x+w,y+h), (255,0,0),thickness=2)
        # The annotated image is displayed in a window named 'Video' using the cv.imshow function.
        cv.imshow('Video', img)

# This line initializes a video capture object (capture) that captures video from the default camera (camera index 0).
capture = cv.VideoCapture(0)

while True: #It starts an infinite loop for continuously capturing and processing video frames.
    isTrue, frame = capture.read() #It reads a frame from the video capture object. The result (isTrue) indicates whether the frame was successfully read, and the frame is stored in the variable frame.
    if isTrue:    #Checks if the frame was successfully read.
        do_Recognize(frame) # Calls the do_Recognize function to perform face recognition on the current video frame.
        if cv.waitKey(20) & 0xFF==ord('d'): # It waits for a key press for 20 milliseconds. If the pressed key is 'd', it breaks out of the loop and ends the program.
            break            
    else:
        break # If the frame was not successfully read, it breaks out of the loop.
 
capture.release() # Releases the video capture object, releasing the video source.

cv.destroyAllWindows() #Closes all OpenCV windows. This is important to clean up resources and close any open windows before ending the program.
