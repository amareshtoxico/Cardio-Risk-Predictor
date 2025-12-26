import pandas as pd
import numpy as np
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup Simple Logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def train_best_model():
    print("🚀 Starting Model Training...")
    try:
        # 2. Load Dataset
        if not os.path.exists('heart.csv'):
            raise FileNotFoundError("heart.csv not found!")

        df = pd.read_csv('heart.csv')
        logger.info(f"Data Loaded. Shape: {df.shape}")

        # 3. Separate Features and Target
        # Assuming the last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # 4. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5. Scaling (CRITICAL: Must match app.py logic)
        # We only scale continuous values, not categorical ones (like sex, cp, etc.)
        scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        scaler = StandardScaler()

        # Fit on training data, transform both
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

        logger.info("Feature Scaling Completed")

        # 6. Train Random Forest (The 'Best' Model)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Random Forest Model Trained")

        # 7. Evaluate Accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"✅ Training Accuracy: {round(train_acc * 100, 2)}%")
        print(f"✅ Test Accuracy:     {round(test_acc * 100, 2)}%")

        # 8. Save the Artifacts
        pickle.dump(model, open('best_model.pkl', 'wb'))
        pickle.dump(scaler, open('scaler.pkl', 'wb'))

        print("\n🎉 Success! 'best_model.pkl' and 'scaler.pkl' have been saved.")
        logger.info("Model and Scaler saved successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Training Failed: {e}")


if __name__ == "__main__":
    train_best_model()