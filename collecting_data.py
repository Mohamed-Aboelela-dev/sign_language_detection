import cv2
import mediapipe as mp
import csv
import os

# Initializing MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Create "data" folder if not exists
if not os.path.exists('data'):
    os.makedirs('data')

# Labels mapping
labels = {
    0: 'Hello',
    1: 'Good',
    2: 'Bad',
    3: 'I love you',
    4: 'Thank you',
    5: 'L',
    6: 'C',
    7: 'Y'
}

print("Available signs to record:")
for key, value in labels.items():
    print(f"{key}: {value}")

# Get user input
sign_id = int(input("Enter sign number (0-7): "))
sign_name = labels[sign_id]

# Start recording
with open('data/sign_data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Append sign ID and save
            landmarks.append(sign_id)
            writer.writerow(landmarks)
            print(f"Recording: {sign_name} - Samples: {file.tell()//64}")
            
        # Display instructions
        cv2.putText(frame, f"Recording: {sign_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "Press 'x' to stop", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow('Sign Language Data Collection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

cap.release()
cv2.destroyAllWindows()