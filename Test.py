import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Loading the model
model = tf.keras.models.load_model('model.h5')

# Initializing MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Categories
labels = ['Hello', 'Good', 'Bad', 'I love you', 'Thank you','L','C','Y']

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Image processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Prediction
        input_data = np.array([landmarks])
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = round(100 * np.max(prediction), 2)

        # Display result with confidence
        text = f"{labels[predicted_class]} ({confidence}%)"
        cv2.putText(frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()