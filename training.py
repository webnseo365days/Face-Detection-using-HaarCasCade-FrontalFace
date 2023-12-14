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


# Initialization of Lists:
# These lists (faces and labels) are initialized to store the detected faces and corresponding labels (person identifiers).
faces = []
labels = []


# Outer Loop Over People:
for person in people: # This loop iterates over a list named people. It seems like people is a list of person identifiers.
    path = os.path.join(DIR,person) #It constructs the path to the directory where the images of the current person are stored. DIR seems to be a variable holding the base directory.
    label = people.index(person) #It assigns a label to the current person based on their index in the people list. This assumes that each person has a unique identifier.
    
    for img in os.listdir(path): #This loop iterates over the images in the directory for the current person.
       
        img_path = os.path.join(path,img)
        img_array = cv.imread(img_path)
        gray= cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        # above 3 lines read each image, converts it to grayscale (gray), which is a common practice for face detection.

     
        # It detects faces in the grayscale image using the Haar Cascade classifier (haar_cascade). Detected faces are represented by rectangles.
        faces_rect= haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors =4 )
        
        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h , x:x+w]
            faces.append(faces_roi)
            labels.append(label)

        # For each detected face, it extracts the region of interest (ROI) from the grayscale image and appends it to the faces list. 
        # The corresponding label (label) is appended to the labels list. This associates each face with its corresponding person label.


#This line creates an instance of the LBPH (Local Binary Pattern Histogram) Face Recognizer using OpenCV. LBPH is a face recognition algorithm.
face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Here, it converts the faces list to a NumPy array. The dtype="object" argument is used because the elements in faces are likely NumPy arrays (grayscale face images).
faces = np.array(faces, dtype="object")

#Similarly, it converts the labels list to a NumPy array. This array holds the corresponding labels for the faces.
labels = np.array(labels)

#This line trains the face recognizer using the provided face images (faces) and their corresponding labels (labels). The train method is used to train the LBPH face recognizer with the given data.
face_recognizer.train(faces,labels)

#It saves the faces array (which contains the face images) to a NumPy binary file named 'faces.npy'. This file can be used later to reload the face images without having to retrain the model
np.save('faces.npy' , faces)

# Similarly, it saves the labels array (which contains the corresponding labels) to a NumPy binary file named 'labels.npy'. This file can be used to reload the labels associated with each face.
np.save('labels.npy' , labels)

# This line saves the trained LBPH face recognizer to a YAML file named 'face_trained.yml'. This file contains the trained model's configuration and parameters, allowing you to load and use the trained model later without retraining.
face_recognizer.save("face_trained.yml")
