import numpy as np
import pandas as pd
import pickle as pkl
import pygeohash as pgh

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from catboost import CatBoostRegressor

# ==================================
# LOAD DATA
# ==================================

data = pd.read_csv("./data/train.csv")

# ==================================
# MISSING VALUES
# ==================================

data["Weather"] = data["Weather"].fillna("unknown")
data["Temperature"] = data["Temperature"].fillna(
    data["Temperature"].median()
)

data["RoadType"] = data["RoadType"].fillna("unknown")
data["timestamp"] = data["timestamp"].fillna("00:00")
data["geohash"] = data["geohash"].fillna("missing")

# ==================================
# BINARY FEATURES
# ==================================

data["LargeVehicles"] = (
    data["LargeVehicles"]
    .fillna("Allowed")
    .map({
        "Not Allowed": 0,
        "Allowed": 1
    })
    .fillna(1)
    .astype(int)
)

data["Landmarks"] = (
    data["Landmarks"]
    .fillna("Yes")
    .map({
        "No": 0,
        "Yes": 1
    })
    .fillna(1)
    .astype(int)
)

# ==================================
# GEOHASH FEATURES
# ==================================

latitudes = []
longitudes = []

for gh in data["geohash"]:
    try:
        lat, lon = pgh.decode(gh)
    except:
        lat, lon = 0, 0

    latitudes.append(lat)
    longitudes.append(lon)

data["latitude"] = latitudes
data["longitude"] = longitudes

# frequency encoding

geo_freq = data["geohash"].value_counts()

data["geo_freq"] = data["geohash"].map(
    geo_freq
)

# ==================================
# TIME FEATURES
# ==================================

data[["hour", "minute"]] = (
    data["timestamp"]
    .str.split(":", expand=True)
)

data["hour"] = data["hour"].astype(int)
data["minute"] = data["minute"].astype(int)

data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)

data["minute_sin"] = np.sin(
    2 * np.pi * data["minute"] / 60
)

data["minute_cos"] = np.cos(
    2 * np.pi * data["minute"] / 60
)

data["is_peak_hour"] = (
    data["hour"]
    .isin([7, 8, 9, 17, 18, 19])
    .astype(int)
)

data["is_night"] = (
    data["hour"]
    .isin([0, 1, 2, 3, 4, 5])
    .astype(int)
)

# ==================================
# INTERACTION FEATURES
# ==================================

data["temp_hour"] = (
    data["Temperature"] * data["hour"]
)

data["temp_peak"] = (
    data["Temperature"] * data["is_peak_hour"]
)

data["lat_lon_interaction"] = (
    data["latitude"] * data["longitude"]
)

# ==================================
# DROP TIMESTAMP
# ==================================

data = data.drop(
    "timestamp",
    axis=1
)

# ==================================
# FEATURES / TARGET
# ==================================

X = data.drop(
    "demand",
    axis=1
)

y = data["demand"]

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================
# TARGET ENCODING
# ==================================

geo_target = (
    X_train.join(y_train)
    .groupby("geohash")["demand"]
    .mean()
)

X_train["geo_target"] = (
    X_train["geohash"]
    .map(geo_target)
)

X_test["geo_target"] = (
    X_test["geohash"]
    .map(geo_target)
)

X_test["geo_target"] = (
    X_test["geo_target"]
    .fillna(geo_target.mean())
)

weather_target = (
    X_train.join(y_train)
    .groupby("Weather")["demand"]
    .mean()
)

X_train["weather_target"] = (
    X_train["Weather"]
    .map(weather_target)
)

X_test["weather_target"] = (
    X_test["Weather"]
    .map(weather_target)
)

X_test["weather_target"] = (
    X_test["weather_target"]
    .fillna(weather_target.mean())
)

# ==================================
# CATEGORICAL FEATURES
# ==================================

cat_features = [
    "geohash",
    "RoadType",
    "Weather"
]

# ==================================
# MODEL
# ==================================

model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.01,
    depth=10,
    l2_leaf_reg=5,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=300,
    verbose=200
)

# ==================================
# TRAIN
# ==================================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ==================================
# PREDICT
# ==================================

preds = model.predict(X_test)

print("\nResults")
print("MAE :", mean_absolute_error(y_test, preds))
print("MSE :", mean_squared_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
print("R2  :", r2_score(y_test, preds))

# ==================================
# SAVE
# ==================================

model.save_model("model.cbm")

with open("temp_median.pkl", "wb") as f:
    pkl.dump(data["Temperature"].median(), f)

with open("feature_columns.pkl", "wb") as f:
    pkl.dump(list(X_train.columns), f)

with open("geo_target_mapping.pkl", "wb") as f:
    pkl.dump(geo_target.to_dict(), f)

with open("weather_target_mapping.pkl", "wb") as f:
    pkl.dump(weather_target.to_dict(), f)

print("Model Saved Successfully")