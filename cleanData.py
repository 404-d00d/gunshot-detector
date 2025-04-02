# RUN 3RD

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load metadata
metaData = pd.read_csv("UrbanSound8K_Full.csv")

# Path to your audio root
audioDir = "D:/archive"

# Prepare lists to hold features and labels
mfccFeatures = []
classLabels = []

# Normalize function (RMS-based to -20 dBFS)
def normalizeAudio(y, targetDBFS=-20):
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y  # silent audio
    scalar = 10 ** (targetDBFS / 20) / rms
    return y * scalar

# Loop over the dataset
for index, row in tqdm(metaData.iterrows(), total=len(metaData)):
    foldName = f"fold{row['fold']}"
    fileName = row["slice_file_name"]
    
    classId = 1 if row["classID"] == 6 else 0
    filePath = os.path.join(audioDir, foldName, fileName)

    try:
        # Load audio (up to 4 seconds)
        y, sr = librosa.load(filePath, duration=4.0)
        
        # Normalize audio volume before MFCC extraction
        y = normalizeAudio(y, targetDBFS=-20)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        mfccMean = np.mean(mfcc.T, axis=0)
        
        mfccFeatures.append(mfccMean)
        classLabels.append(classId)

    except Exception as error:
        print(f"Error processing {filePath}: {error}")

# Convert to NumPy arrays
xData = np.array(mfccFeatures)
yData = np.array(classLabels)

# Save features
np.save("xDataMfcc.npy", xData)
np.save("yDataLabels.npy", yData)