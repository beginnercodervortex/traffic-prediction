import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("./data/train.csv")

# =========================
# HANDLE MISSING VALUES
# =========================

# categorical columns
categorical_cols = [
    "geohash",
    "timestamp",
    "RoadType",
    "Weather"
]

for col in categorical_cols:
    data[col] = (
        data[col]
        .fillna("missing")
        .astype(str)
    )

# numerical column
temp_median = data["Temperature"].median()

data["Temperature"] = (
    data["Temperature"]
    .fillna(temp_median)
)

# binary columns
data["LargeVehicles"] = (
    data["LargeVehicles"]
    .fillna("Allowed")
    .map({
        "Not Allowed": 0,
        "Allowed": 1
    })
)

data["Landmarks"] = (
    data["Landmarks"]
    .fillna("Yes")
    .map({
        "No": 0,
        "Yes": 1
    })
)

# if dataset contains other values
data["LargeVehicles"] = (
    data["LargeVehicles"]
    .fillna(1)
    .astype(int)
)

data["Landmarks"] = (
    data["Landmarks"]
    .fillna(1)
    .astype(int)
)

# =========================
# FINAL NULL CHECK
# =========================

print("\nMissing Values:")
print(data.isnull().sum())

# =========================
# FEATURES / TARGET
# =========================

X = data.drop("demand", axis=1)
y = data["demand"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CATEGORICAL FEATURES
# =========================

cat_features = [
    "geohash",
    "timestamp",
    "RoadType",
    "Weather"
]

# =========================
# MODEL
# =========================

model = LGBMRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

# =========================
# TRAIN
# =========================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# =========================
# PREDICT
# =========================

preds = model.predict(X_test)

# =========================
# METRICS
# =========================

print("\nResults")
print("MAE :", mean_absolute_error(y_test, preds))
print("MSE :", mean_squared_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("R2  :", r2_score(y_test, preds))

# =========================
# SAVE MODEL
# =========================

with open("model.pkl", "wb") as f:
    pkl.dump(model, f)

with open("temp_median.pkl", "wb") as f:
    pkl.dump(temp_median, f)

with open("feature_columns.pkl", "wb") as f:
    pkl.dump(list(X.columns), f)

print("\nModel Saved Successfully")