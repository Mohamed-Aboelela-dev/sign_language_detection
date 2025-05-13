# 🤟 Sign Language Recognition

This project aims to recognize sign language gestures using computer vision and deep learning. It consists of three main parts:

1. **Data Collection** 📷: Capturing sign language gestures and saving the data.
2. **Model Training** 🧠: Training a machine learning model using the collected data.
3. **Model Inference** 🔍: Using the trained model to classify new hand gestures in real-time.

## Project Structure

The project is organized as follows:
```
/project-directory
│
├── collecting_data.py # 📷 Script for collecting sign language data.
├── Test.py # 🧪 Script for testing the trained model with live camera input.
├── Train.py # 🏋️‍♀️ Script for training the model with the collected data.
├── requirements.txt # 📦 Required dependencies for the project.
└── data/ # 📁 Directory where sign language gesture data is stored.
└── sign_data.csv # 📝 CSV file containing the collected hand landmarks data.
```

## Requirements 🛠️

To set up and run the project, you need to install the required dependencies. You can do so using `pip`:
```bash
pip install -r requirements.txt
```

The dependencies are listed in the `requirements.txt` file:

- `pip~=25.0.1`
- `wheel~=0.45.1`
- `PySocks~=1.7.1`
- `protobuf~=3.20.3`
- `setuptools~=75.3.2`
- `opencv-python~=4.11.0.86`
- `mediapipe~=0.10.11`
- `numpy~=1.24.3`
- `tensorflow~=2.13.0`
- `pandas~=2.0.3`
- `scikit-learn~=1.3.2`

## Data Collection 📸

To collect data for different sign language gestures, run the `collecting_data.py` script. It uses the webcam to capture hand gestures and stores the data (hand landmarks) in a CSV file.

### Running Data Collection:

1. Run the `collecting_data.py` script:
```bash
python collecting_data.py
```

2. The script will ask you to choose a sign gesture by entering its corresponding number (0-7).
3. Perform the gesture in front of your camera. The landmarks for each gesture will be recorded and stored in `data/sign_data.csv`.
4. Press 'x' to stop recording.

The available gestures are:

- 0: ✋ Hello
- 1: 👍 Good
- 2: 👎 Bad
- 3: 🤟 I love you
- 4: 🙏 Thank you
- 5: L
- 6: C
- 7: Y

## Model Training 🏋️‍♀️

To train the machine learning model, run the `Train.py` script. It uses the collected gesture data (`sign_data.csv`) to train a neural network for gesture classification.

### Running Model Training:

1. Ensure you have collected enough data for training.
2. Run the `Train.py` script:
```bash
python Train.py
```

3. The script will train a neural network model and save it as `model.h5` in the project directory.

## Model Inference 🧑‍🏫

To use the trained model for real-time sign language recognition, run the `Test.py` script. It will use your webcam to classify gestures in real-time.

### Running Model Inference:

1. Make sure the model is trained and saved as `model.h5`.
2. Run the `Test.py` script:
```bash
python Test.py
```

3. Perform a gesture in front of the camera. The model will predict the gesture and display the predicted label along with the confidence percentage on the screen.
4. Press 'x' to stop the inference.
