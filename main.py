import cv2
import sys
import pickle
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# testing images
ryan = "model_images/ryan_reynolds/download.jpg"

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

image_counter = 0

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #frame = cv2.imread(ryan)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 50:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
            image_counter += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        """
        img_item = str(image_counter)+".png"
        save_path = "model_images/" + labels[id_]
        cv2.imwrite(os.path.join(save_path, img_item), roi_color)
        print(os.path.join(save_path, img_item))
        """

    # Display the resulting frame
    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
