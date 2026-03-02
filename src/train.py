# train.py

from preprocess import load_data, build_preprocessor
from model import build_model
import numpy as np
import joblib

data = load_data("data/cover_data.csv")

X = data.drop(columns=["class"])
y = data["class"]

preprocessor = build_preprocessor(X)
X_processed = preprocessor.fit_transform(X)

model = build_model(X_processed.shape[1], num_classes=7)

history = model.fit(
    X_processed,
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

model.save("model.h5")
joblib.dump(preprocessor, "preprocessor.pkl")