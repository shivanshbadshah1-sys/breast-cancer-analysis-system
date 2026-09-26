import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))

# Use only the 4 features the app expects
FEATURES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
data = data.rename(columns={
    "radius_mean": "radius_mean",
    "texture_mean": "texture_mean",
    "perimeter_mean": "perimeter_mean",
    "area_mean": "area_mean"
})

X = data[FEATURES]
y = data["diagnosis"].map({"M": 1, "B": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
print("✅ Model Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved")

# Bar chart
labels = ["Benign", "Malignant"]
values = [int((y == 0).sum()), int((y == 1).sum())]
plt.figure(figsize=(5, 4))
plt.bar(labels, values, color=["#00c9a7", "#ff5252"])
plt.title("Breast Cancer Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "static", "chart.png"))
plt.close()
print("📊 chart.png saved")
