# 🌲 Forest Cover Type Classification --- Deep Learning Pipeline

## 📌 Overview

This project builds and evaluates a deep neural network to classify
forest cover types using cartographic and environmental features. The
goal is to develop a reproducible, production-oriented machine learning
pipeline with structured preprocessing, regularization, and validation
strategies.

The project demonstrates:

-   End-to-end ML workflow
-   Structured data preprocessing (scaling + encoding)
-   Model regularization and overfitting control
-   Proper train/validation/test separation
-   Reproducible environment setup
-   Version-controlled ML development

------------------------------------------------------------------------

## 🗂 Dataset

**Source:** UCI Forest Cover Type Dataset\
**Samples:** \~54,000\
**Features:** 54 cartographic variables\
**Target:** Forest Cover Type (multi-class classification)

Feature types include:

-   Continuous numerical variables (elevation, slope, distances)
-   Binary indicator variables
-   Categorical attributes (encoded)

------------------------------------------------------------------------

## 🧠 Problem Formulation

Multi-class classification problem:

Predict the correct forest cover type based on environmental and
geographical attributes.

Formally:

f(X) → y

Where: - X = structured feature vector - y ∈ {1, 2, ..., 7}

------------------------------------------------------------------------

## 🔄 Data Pipeline

### 1️⃣ Train / Validation / Test Split

-   80% Training
-   10% Validation
-   10% Testing
-   Fixed random seed for reproducibility

### 2️⃣ Preprocessing

Implemented using `sklearn.ColumnTransformer`:

-   `StandardScaler` for numeric features
-   `OneHotEncoder` for categorical features
-   Fit on training data only (no leakage)

------------------------------------------------------------------------

## 🏗 Model Architecture

Deep feedforward neural network built using TensorFlow (Keras):

Input Layer\
→ Dense(128, ReLU)\
→ Dropout(0.2)\
→ Dense(64, ReLU)\
→ Dense(7, Softmax)

### Regularization Strategy

-   Dropout (0.2) to reduce overfitting
-   EarlyStopping on validation loss
-   Validation monitoring with restore_best_weights=True

------------------------------------------------------------------------

## ⚙️ Training Configuration

-   Optimizer: Adam
-   Loss: Sparse Categorical Crossentropy
-   Batch Size: 32
-   EarlyStopping (patience=5)

------------------------------------------------------------------------

## 📊 Model Performance

| Metric | Validation | Test |
|--------|-----------:|-----:|
| Loss (MSE) | 0.3933 | 0.3938 |
| MAE | 0.3618 | 0.3615 |

Training and validation curves were monitored to ensure proper
generalization and prevent overfitting.

------------------------------------------------------------------------

## 🧪 Reproducibility

### Install dependencies

pip install -r requirements.txt

### Train model

python src/train.py

### Evaluate model

python src/evaluate.py

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python 3.12
-   TensorFlow / Keras
-   Scikit-learn
-   NumPy
-   Pandas
-   Matplotlib
-   Git (SSH authentication workflow)

------------------------------------------------------------------------

## 🚀 Engineering Highlights

-   Clean separation of preprocessing and modeling
-   No data leakage during scaling/encoding
-   Modular structure (preprocess / train / evaluate)
-   Version-controlled development
-   SSH-based GitHub authentication
-   Reproducible environment via requirements.txt

------------------------------------------------------------------------

## 🔮 Potential Improvements

-   Hyperparameter tuning (Optuna / KerasTuner)
-   Cross-validation instead of single split
-   Feature importance analysis
-   Model deployment via FastAPI / Streamlit
-   Experiment tracking (MLflow)

------------------------------------------------------------------------

## 📬 Author

Yunping Cai\
ML Engineer \| Data Engineering \| Applied AI
