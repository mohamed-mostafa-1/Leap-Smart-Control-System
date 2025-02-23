# Hand Gesture Recognition using IMU Sensors

## Overview
This project implements a real-time hand gesture recognition system using IMU (Inertial Measurement Unit) sensors. The system processes real-time motion data and classifies hand gestures using a trained deep learning model. The main components of this project include data acquisition, preprocessing, model training, and real-time inference.

## Features
- **Real-time gesture recognition** using MPU-6050 and ESP32
- **Deep learning-based classification** with TensorFlow/Keras
- **Modular design** with separate components for data acquisition, preprocessing, training, and prediction
- **Integration with IoT devices** for wireless communication
- **Python-based implementation** for flexibility and extensibility

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PySerial (for serial communication with ESP32)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Project Structure
```
hand_gesture_recognition/
│── data_acquisition.py    # Collects data from IMU sensors
│── data_preprocessing.py  # Prepares data for training
│── model_training.py      # Trains the deep learning model
│── real_time_prediction.py # Runs real-time inference
│── utils.py              # Helper functions
│── models/               # Stored trained models
│── data/                 # Raw and processed data
│── README.md             # Project documentation
│── requirements.txt      # Required dependencies
```

## Usage
### 1. Data Collection
Run the following script to collect sensor data:
```sh
python data_acquisition.py
```

### 2. Data Preprocessing
Convert raw sensor data into a suitable format for training:
```sh
python data_preprocessing.py
```

### 3. Model Training
Train the deep learning model with processed data:
```sh
python model_training.py
```

### 4. Real-Time Prediction
Use the trained model to classify gestures in real time:
```sh
python real_time_prediction.py
```

## Troubleshooting
### TensorFlow Import Error (Protobuf issue)
If you encounter errors related to `protobuf`, try:
```sh
pip install protobuf==3.20.*
```
Or set:
```sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Contributing
Feel free to submit pull requests or report issues. Any contributions to improving this project are welcome!


