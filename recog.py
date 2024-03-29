import cv2
import os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX  # Creates a font
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
        100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
        if conf < 100:
            nbr_predicted = 'Ryan!!!!!!'
        else:
            nbr_predicted = "Not Ryan >:(((((("
        cv2.putText(im, str(nbr_predicted),
                    (x, y+h), font, 1.1, (255, 255, 255))  # Draw the text
    cv2.imshow('RYAN!!!!!!', im)
    cv2.waitKey(1)
