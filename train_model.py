import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("wine_data.csv")

# Convert to classification
def quality_label(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

df["label"] = df["quality"].apply(quality_label)

X = df.drop(["quality", "label"], axis=1)
y = df["label"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ✅ SMALL & FAST MODEL (IMPORTANT FOR SIZE)
model = RandomForestClassifier(
    n_estimators=30,   # smaller = smaller file
    max_depth=8,       # limits size
    random_state=42
)

model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model & scaler saved successfully")