import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained model for emotion detection (example with deepface)
emotion_model = DeepFace.build_model("Emotion")

# Define a dictionary to map numerical predictions to emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Start video capture from default camera
cap = cv2.VideoCapture(0)

# Initialize variables for smoothing emotion prediction and storing all detected emotions
prev_emotion = None
count = 0
all_emotions = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face ROI to fit the input size of the emotion model
        resized_face = cv2.resize(face_roi, (48, 48))

        # Predict the emotion using the emotion model
        emotion_prediction = emotion_model.predict(resized_face[np.newaxis, :, :, :])

        # Get the emotion label
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Smoothing mechanism
        if prev_emotion == emotion_label:
            count += 1
            if count >= 5:  # Adjust the count threshold as needed
                all_emotions.append(emotion_label)
                count = 0
        else:
            prev_emotion = emotion_label
            count = 0

    # Calculate the most frequent emotion from all detected emotions
    most_common_emotion = None
    if all_emotions:
        most_common_emotion = Counter(all_emotions).most_common(1)[0][0]

    # Display the detected emotion label on the screen
    if prev_emotion:
        cv2.putText(frame, f"Detected Emotion: {prev_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the most frequent emotion label on the screen
    if most_common_emotion:
        cv2.putText(frame, f"Most Frequent Emotion: {most_common_emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()