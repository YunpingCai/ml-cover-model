# src/train.py
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocess import load_data, build_preprocessor
from model import build_model

# --- paths (repo root assumed) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cover_data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# --- load ---
data = load_data(DATA_PATH)

TARGET_COL = "class"   # change if needed
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# --- split train/val/test (80/10/10) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- preprocess (fit on train only!) ---
preprocessor = build_preprocessor(X_train)
x_train_processed = preprocessor.fit_transform(X_train)
x_val_processed = preprocessor.transform(X_val)
x_test_processed = preprocessor.transform(X_test)

# --- build model ---
num_classes = int(np.unique(y).shape[0])
model = build_model(x_train_processed.shape[1], num_classes=num_classes)

# --- train ---
history = model.fit(
    x_train_processed, y_train,
    validation_data=(x_val_processed, y_val),
    epochs=50,
    batch_size=32
)

# --- save model + preprocessor ---
model_path = MODELS_DIR / "forest_cover_model.keras"
prep_path = MODELS_DIR / "preprocessor.pkl"

model.save(model_path)
joblib.dump(preprocessor, prep_path)

# --- evaluate ---
test_metrics = model.evaluate(x_test_processed, y_test, verbose=0)
print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))

# --- save training curve ---
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Model Loss")
plt.legend()
plt.savefig(REPORTS_DIR / "training_curve.png", dpi=200, bbox_inches="tight")
plt.close()