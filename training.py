import cv2 as cv         # cv2 is used for image processing
import os                # os is used for directory related processing
import numpy as np       # numpy is very famous for handling arrays / matrices 

# following is list of person whose faces you want to detect. There is Faces directory. Open it and you will find two more directories name train and val
# in each directory you will find 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling' directories. you can add more for more persons.
# add pictures in each directory
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


 
p = []  # Create an empty list named 'p'
# Iterate over the contents of the directory "Faces\train"
for i in os.listdir(r"Faces\train"):
    p.append(i)  # Append each item (file or directory name) to the list 'p'
# The r before the string (e.g., r"Faces\train") denotes a raw string literal, which is used to interpret backslashes as literal characters rather than escape characters. 
# This is often used with file paths to avoid issues with backslashes being interpreted as escape characters.
# p.append(i): Inside the loop, each item (i) is appended to the list p. 
# This means that after the loop completes, p will contain all the file or directory names present in the "Faces\train" directory.

DIR = r"Faces\train"

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# This line of code is using OpenCV (cv2) to create a Haar Cascade Classifier for detecting frontal faces in images. Let's break it down:
# cv.data.haarcascades: This is a path to the directory containing Haar Cascade XML files provided by OpenCV. 
# These XML files define the features and parameters for object detection using Haar cascades.
# 'haarcascade_frontalface_default.xml': This is the specific Haar Cascade XML file for detecting frontal faces. 
# The file contains information about the patterns and features that constitute a frontal face.
# cv.CascadeClassifier(...): This part initializes a CascadeClassifier object, which is a classifier based on Haar cascades. 
# It takes the complete path to the XML file as an argument.



faces = []
labels = []

for person in people:
    path = os.path.join(DIR,person)
    label = people.index(person)
    
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        
        img_array = cv.imread(img_path)
        gray= cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        
        faces_rect= haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors =4 )
        
        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h , x:x+w]
            faces.append(faces_roi)
            labels.append(label)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
faces = np.array(faces, dtype="object")
labels = np.array(labels)
face_recognizer.train(faces,labels)
np.save('faces.npy' , faces)
np.save('labels.npy' , labels)
face_recognizer.save("face_trained.yml")
