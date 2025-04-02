# RUN 2ND

import os
import pandas as pd

# Folder with augmented gunshots
augmentedFolder = "D:/archive/fold11"

# List all .wav files
fileList = [f for f in os.listdir(augmentedFolder) if f.endswith(".wav")]

# Build a list of rows
rows = []
for fileName in fileList:
    baseName = os.path.splitext(fileName)[0]  # e.g. 100263-6-0-1
    parts = baseName.split('-')

    if len(parts) != 4:
        print(f"Skipping unexpected file name: {fileName}")
        continue

    fsID, classID, sliceID, instance = parts

    row = {
        "slice_file_name": fileName,
        "fsID": int(fsID),
        "start": 0.0,          # placeholder
        "end": 4.0,            # max clip duration
        "salience": 1,         # placeholder
        "fold": 11,
        "classID": int(classID),
        "class": "gun_shot"
    }

    rows.append(row)

# Create DataFrame
augmentedDF = pd.DataFrame(rows)

# Save to CSV
outputPath = "UrbanSound8K_Augmented.csv"
augmentedDF.to_csv(outputPath, index=False)

print(f"Augmented metadata saved to: {outputPath}")

# Load both CSVs
originalDF = pd.read_csv("UrbanSound8K.csv")
augmentedDF = pd.read_csv("UrbanSound8K_Augmented.csv")

# Combine
mergedDF = pd.concat([originalDF, augmentedDF], ignore_index=True)

# Save to a new file
mergedDF.to_csv("UrbanSound8K_Full.csv", index=False)
print("Full dataset with original + augmented saved to UrbanSound8K_Full.csv")
