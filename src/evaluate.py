# evaluate.py

import joblib
import tensorflow as tf
from sklearn.metrics import classification_report

model = tf.keras.models.load_model("model.h5")
preprocessor = joblib.load("preprocessor.pkl")

# load test data...
# transform
# predict
# print metrics