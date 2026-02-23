import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    # 1. Load Data
    data_path = 'data/posture_data.csv'
    if not os.path.exists(data_path):
        print("Error: posture_data.csv not found!")
        return

    df = pd.read_csv(data_path)
    X = df[['score']]  # Features
    y = df['label']    # Target (0 or 1)

    # 2. Split Data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize and Train Random Forest
    # We use Random Forest because it handles "noisy" sensor data well
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Training Complete! Accuracy: {round(acc * 100, 2)}%")

    # 5. Save the Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model.pkl')
    print("Model saved as 'models/stress_model.pkl'")

if __name__ == "__main__":
    train_model()