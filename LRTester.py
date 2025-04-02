import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn
import os

# Import your custom Logistic Regression
from LogisticRegression import BinaryLogisticRegression

# Load your MFCC features and labels
X = np.load("xDataMfcc.npy")
y = np.load("yDataLabels.npy")

# Split the data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data (WORSENS PERFORMANCE - DO NOT USE)
scaler = StandardScaler()
XTrainScale = scaler.fit_transform(XTrain)
XTestScale = scaler.transform(XTest)  

# Initialize and train your logistic regression model
model = BinaryLogisticRegression()
model.fit(XTrain, yTrain, alpha=0.0001, epochs=500, eps=1e-3, update=50)

# Make predictions
predictions = model.predict(XTest)

# Print classification results
print("Classification Report:")
print(classification_report(yTest, predictions, target_names=["Non-Gunshot", "Gunshot"]))

# Confusion matrix
confMat = confusion_matrix(yTest, predictions)
print("Confusion Matrix:\n", confMat)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sn.heatmap(confMat, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Gunshot", "Gunshot"], yticklabels=["Non-Gunshot", "Gunshot"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Gunshot Detector")
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
	preds = model.predict(features, 0.1).flatten()

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