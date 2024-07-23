import cv2
import os
import numpy as np

dataPath = "C:/Users/Daniel/Desktop/datos_entrenamiento"
peopleList = os.listdir(dataPath)

print(peopleList)

labels = []
faces_data = []

label = 0

for nameDir in peopleList:
    personPath = dataPath +'/'+ nameDir
    #print("reading images")

    for filename in os.listdir(personPath):
        labels.append(label)
        faces_data.append(cv2.imread(personPath+'/'+filename,0))
     
    label += 1

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("training ...")

face_recognizer.train(faces_data, np.array(labels))

print("saving model ...")

face_recognizer.write("model.xml")

print("model saved")

