import numpy as np

def compute_features(df):
    df['acc_magnitude'] = np.sqrt(df['AcX']**2 + df['AcY']**2 + df['AcZ']**2)
    df['gyro_magnitude'] = np.sqrt(df['GyX']**2 + df['GyY']**2 + df['GyZ']**2)
    
    # Compute rolling mean and std for better feature extraction
    window_size = 5
    for col in ['AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ']:
        df[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
        df[f'{col}_std'] = df[col].rolling(window=window_size).std()

    df.dropna(inplace=True)
    return df
