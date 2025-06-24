import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

input_path = "MLProject/heartdataset/heart.csv"
df = pd.read_csv(input_path)

X = df.drop("condition", axis=1)
y = df["condition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df["condition"] = y_train.reset_index(drop=True)
train_df.to_csv("heart_cleveland_train.csv", index=False)

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["condition"] = y_test.reset_index(drop=True)
test_df.to_csv("heart_cleveland_test.csv", index=False)

print("✅ Preprocessing selesai. Dataset disimpan.")
