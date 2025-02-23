import firebase_admin # type: ignore
from firebase_admin import credentials, db # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

cred = credentials.Certificate("D:\My Learning\College Material\Graduation project\HGR\hand_gesture_recognition\leap.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://leap-smart-band-default-rtdb.firebaseio.com/"})

model = tf.keras.models.load_model("gesture_model.h5")

def fetch_real_time_data():
    ref = db.reference("/IMU_Data")
    data = ref.get()
    imu_data = np.array([data['0']['Imu0_linear_accleration_x'],
                         data['0']['Imu0_linear_accleration_y'],
                         data['0']['Imu0_linear_accleration_z'],
                         data['0']['Imu0_orientation_x'],
                         data['0']['Imu0_orientation_y'],
                         data['0']['Imu0_orientation_z']])
    
    imu_data = imu_data.reshape(1, 1, -1)  # Reshape for LSTM input
    prediction = np.argmax(model.predict(imu_data))
    print(f"Predicted Gesture: {prediction}")

if __name__ == "__main__":
    while True:
        fetch_real_time_data()
