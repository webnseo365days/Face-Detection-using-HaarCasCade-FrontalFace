import cv2 as cv         # cv2 is used for image processing
import os                # os is used for directory related processing
import numpy as np       # numpy is very famous for handling arrays / matrices 

# following is list of person whose faces you want to detect. There is Faces directory. Open it and you will find two more directories name train and val
# in each directory you will find 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling' directories. you can add more for more persons.
# add pictures in each directory
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

p =[]
for i in os.listdir(r"Faces\train"):
    p.append(i)

DIR = r"Faces\train"

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
