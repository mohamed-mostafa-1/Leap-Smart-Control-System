import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=5, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_imu_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Normalize data
    for col in ['AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Apply low-pass filter
    for col in ['AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ']:
        df[col] = butter_lowpass_filter(df[col])

    return df
