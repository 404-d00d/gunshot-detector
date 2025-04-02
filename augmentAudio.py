# RUN FIRST
import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, lfilter

def bandpassFilter(data, sr, lowcut=100, highcut=6000, order=6):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def softLimiter(y, threshold=0.6):
    return np.tanh(y / threshold)

# Load metadata
metaData = pd.read_csv("UrbanSound8K.csv")

audioDir = "D:/archive"
outputDir = "D:/archive/fold11"
os.makedirs(outputDir, exist_ok=True)

gunshotData = metaData[metaData["classID"] == 6]

for index, row in tqdm(gunshotData.iterrows(), total=len(gunshotData)):
    foldName = f"fold{row['fold']}"
    fileName = row["slice_file_name"]
    filePath = os.path.join(audioDir, foldName, fileName)

    try:
        y, sr = librosa.load(filePath, duration=4.0)
        parts = os.path.splitext(fileName)[0].split('-')
        fsID, classID, sliceID, _ = parts

        # Aug 1–7
        noise = np.random.normal(0, 0.005, y.shape)
        y1 = y + noise
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-1.wav"), y1, sr)

        y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-2.wav"), y2, sr)

        y3 = librosa.effects.time_stretch(y, rate=0.9)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-3.wav"), y3, sr)

        y4 = librosa.effects.pitch_shift(y + np.random.normal(0, 0.005, y.shape), sr=sr, n_steps=-2)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-4.wav"), y4, sr)

        y5 = librosa.effects.time_stretch(y + np.random.normal(0, 0.005, y.shape), rate=0.9)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-5.wav"), y5, sr)

        y6 = librosa.effects.time_stretch(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2), rate=0.9)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-6.wav"), y6, sr)

        y7 = librosa.effects.time_stretch(librosa.effects.pitch_shift(y + np.random.normal(0, 0.005, y.shape), sr=sr, n_steps=-2), rate=0.9)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-7.wav"), y7, sr)

        # Volume Drop (8–15)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-8.wav"), y1 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-9.wav"), y2 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-10.wav"), y3 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-11.wav"), y4 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-12.wav"), y5 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-13.wav"), y6 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-14.wav"), y7 * 0.3, sr)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-15.wav"), y * 0.3, sr)

        # Clipping (16–21)
        y16 = np.clip(y, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-16.wav"), y16, sr)

        y17 = np.clip(y1, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-17.wav"), y17, sr)

        y18 = np.clip(y2, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-18.wav"), y18, sr)

        y19 = np.clip(y3, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-19.wav"), y19, sr)

        y20 = np.clip(y4, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-20.wav"), y20, sr)

        y21 = np.clip(y7, -0.4, 0.4)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-21.wav"), y21, sr)

        # Clipping + Boosted (22–27)
        y22 = np.clip(y * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-22.wav"), y22, sr)

        y23 = np.clip(y1 * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-23.wav"), y23, sr)

        y24 = np.clip(y2 * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-24.wav"), y24, sr)

        y25 = np.clip(y3 * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-25.wav"), y25, sr)

        y26 = np.clip(y4 * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-26.wav"), y26, sr)

        y27 = np.clip(y7 * 1.2, -1.0, 1.0)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-27.wav"), y27, sr)

        # Soft Limiting (28–33)
        y28 = softLimiter(y, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-28.wav"), y28, sr)

        y29 = softLimiter(y1, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-29.wav"), y29, sr)

        y30 = softLimiter(y2, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-30.wav"), y30, sr)

        y31 = softLimiter(y3, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-31.wav"), y31, sr)

        y32 = softLimiter(y4, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-32.wav"), y32, sr)

        y33 = softLimiter(y5, threshold=0.6)
        sf.write(os.path.join(outputDir, f"{fsID}-{classID}-{sliceID}-33.wav"), y33, sr)


    except Exception as e:
        print(f"Error processing {filePath}: {e}")


acData = metaData[metaData["classID"] == 0]  # Air Conditioner

for index, row in tqdm(acData.iterrows(), total=len(acData)):
    foldName = f"fold{row['fold']}"
    fileName = row["slice_file_name"]
    filePath = os.path.join(audioDir, foldName, fileName)

    try:
        y, sr = librosa.load(filePath, duration=4.0)

        for i, factor in enumerate([0.0, 0.1, 0.2, 0.3]):
            quiet = y * factor
            noise = np.random.normal(0, 0.002*factor, quiet.shape)
            quietNoisy = quiet + noise
            clipped = np.clip(quietNoisy, -0.4, 0.4)

            baseName = os.path.splitext(fileName)[0]
            newName = f"{baseName}0{i}.wav"
            sf.write(os.path.join(outputDir, newName), clipped, sr)

    except Exception as e:
        print(f"Error processing {filePath}: {e}")