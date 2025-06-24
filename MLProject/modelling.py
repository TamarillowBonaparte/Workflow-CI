import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Prediction")

train = pd.read_csv("heart_cleveland_train.csv")
test = pd.read_csv("heart_cleveland_test.csv")

X_train = train.drop("condition", axis=1)
y_train = train["condition"]
X_test = test.drop("condition", axis=1)
y_test = test["condition"]

# Tidak perlu start_run
mlflow.sklearn.autolog()

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Simpan model ke outputs
joblib.dump(model, "MLProject/trained_model.pkl")

