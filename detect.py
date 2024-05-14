import cv2
import numpy as np
import os
import sqlite3

# Load the pre-trained face detector
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Start capturing video from the default camera
cam = cv2.VideoCapture(0)

# Load the trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")


# Function to get the profile of a student from the database using their ID
def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


while True:
    # Capture frame-by-frame
    ret, img = cam.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Recognize the face and get the ID and confidence level
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        # Get the profile information of the student with the recognized ID
        profile = get_profile(id)
        # If the confidence level is below a certain threshold and the profile matches the detected face, print the profile information
        if profile is not None and conf < 100:
            cv2.putText(img, "Name:" + str(profile[1]), (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Dob:" + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Id:" + str(profile[0]), (x, y + h + 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 127), 2)
        else:
            # If the face is not recognized, print "Unrecognized"
            cv2.putText(img, "Unrecognized", (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("FACE", img)
    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cam.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
