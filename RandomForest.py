import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# === Load MFCC Features ===
X = np.load("xDataMfcc.npy")      # Shape: (n_samples, n_features)
y = np.load("yDataLabels.npy")    # Binary: 0 = Non-Gunshot, 1 = Gunshot

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train Random Forest ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
predictions = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, predictions, target_names=["Non-Gunshot", "Gunshot"]))

print("=== Confusion Matrix ===")
confMat = confusion_matrix(y_test, predictions)
print(confMat)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sn.heatmap(confMat, annot=True, fmt='d', cmap='Blues',
           xticklabels=["Non-Gunshot", "Gunshot"],
           yticklabels=["Non-Gunshot", "Gunshot"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Random Forest Gunshot Detector")
plt.show()

# List your real-world files
realFeatureFiles = [
    #"D:/bodycam_splits/Nashville School Shooting Body Camera Footage_mfcc.npy",
    "D:/bodycam_splits/Police Save Woman From Predator Ex-Husband_mfcc.npy",
    "D:/bodycam_splits/Predator Pulls Gun During Sting_mfcc.npy",
    "D:/bodycam_splits/Cop Uses MP7 To Shred Suspect_mfcc.npy"
]

# Predict on each and save results
for path in realFeatureFiles:
    features = np.load(path)
    preds = model.predict(features).flatten()

    labels = ["Gunshot" if p == 1 else "Non-Gunshot" for p in preds]

    gunshotDetected=0
    print(f"\n=== Predictions for {os.path.basename(path)} ===")
    for i, label in enumerate(labels):
        chunkLength = 1
        startTime = i * chunkLength
        endTime = startTime + chunkLength
        print(f"Chunk {i} ({startTime}s–{endTime}s): {label}")
        if label == "Gunshot":
            gunshotDetected+=1

    print(f"{gunshotDetected}(s) detected out of {len(labels)} chunks")

    # Save to CSV
    df = pd.DataFrame({
        "chunk": list(range(len(labels))),
        "prediction": preds,
        "label": labels
    })

    baseName = os.path.splitext(os.path.basename(path))[0]
    df.to_csv(f"{baseName}_predictions.csv", index=False)
    print(f"Saved predictions to {baseName}_predictionsLR.csv")