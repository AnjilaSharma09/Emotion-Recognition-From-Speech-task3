# Emotion-Recognition-From-Speech-task3
code:
pip install librosa scikit-learn numpy pandas
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Define a function to extract features from an audio file
def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Compute the mean of the MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean

# Define a function to prepare the dataset
def prepare_dataset(directory):
    features = []
    labels = []
    
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            
            # Extract features
            feature_vector = extract_features(file_path)
            
            # Assume the emotion label is in the filename (e.g., "happy_001.wav")
            label = filename.split('_')[0]
            
            features.append(feature_vector)
            labels.append(label)
    
    return np.array(features), np.array(labels)

# Load dataset
directory = 'path/to/your/audio/files'  # Update this to the path where your audio files are stored
X, y = prepare_dataset(directory)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict emotion from a new audio file
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Example usage
new_audio_file = 'path/to/your/new_audio_file.wav'
predicted_emotion = predict_emotion(new_audio_file)
print(f'Predicted Emotion: {predicted_emotion}')
