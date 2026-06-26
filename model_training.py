import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset (UPDATE PATH IF NEEDED)
df = pd.read_excel("Dataset/flood dataset.xlsx")
# Features and target
X = df.drop("flood", axis=1)
y = df["flood"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save
os.makedirs("Models", exist_ok=True)

joblib.dump(model, "Models/model.pkl")
joblib.dump(scaler, "Models/scaler.pkl")

print("✅ Model trained and saved successfully!")