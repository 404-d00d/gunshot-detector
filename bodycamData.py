import os
import librosa
import soundfile as sf
import numpy as np
import re
import pandas as pd

# === Settings ===
inputFiles = [
    #"D:/Nashville School Shooting Body Camera Footage.wav",
    "D:/Police Save Woman From Predator Ex-Husband.wav",
    "D:/Predator Pulls Gun During Sting.wav",
    "D:/Cop Uses MP7 To Shred Suspect.wav"
]

baseOutputDir = "D:/bodycam_splits"
chunkDuration = 1.0  # seconds

# Loop over each input audio file
for inputPath in inputFiles:
    # Use filename (without extension) to name output folder
    baseName = os.path.splitext(os.path.basename(inputPath))[0]
    outputDir = os.path.join(baseOutputDir, baseName)
    os.makedirs(outputDir, exist_ok=True)

    # Clear the output directory first
    if os.path.exists(outputDir):
        for filename in os.listdir(outputDir):
            filePath = os.path.join(outputDir, filename)
            try:
                if os.path.isfile(filePath):
                    os.remove(filePath)
            except Exception as e:
                print(f"Error deleting {filePath}: {e}")


    # === Load audio ===
    y, sr = librosa.load(inputPath, sr=None)
    totalDuration = librosa.get_duration(y=y, sr=sr)
    numChunks = int(totalDuration // chunkDuration)

    # === Split into chunks ===
    for i in range(numChunks):
        start = int(i * chunkDuration * sr)
        end = int((i + 1) * chunkDuration * sr)
        chunk = y[start:end]
        fileName = f"{i}_bodycamSecond.wav"
        sf.write(os.path.join(outputDir, fileName), chunk, sr)

    print(f"{baseName}: {numChunks} chunks saved to {outputDir}")

allFeatures = []

for inputPath in inputFiles:
    baseName = os.path.splitext(os.path.basename(inputPath))[0]
    outputDir = os.path.join(baseOutputDir, baseName)

    # Sort files numerically
    fileList = sorted(
        [f for f in os.listdir(outputDir) if f.endswith(".wav")],
        key=lambda x: int(re.search(r"(\d+)", x).group(1))
    )

    mfccFeatures = []

    for file in fileList:
        filePath = os.path.join(outputDir, file)
        y, sr = librosa.load(filePath, duration=chunkDuration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        mfccMean = np.mean(mfcc.T, axis=0)
        mfccFeatures.append(mfccMean)

    allFeatures.append((baseName, mfccFeatures))
    # Save each feature set as .npy
    np.save(os.path.join(baseOutputDir, f"{baseName}_mfcc.npy"), np.array(mfccFeatures))
