# Titanic Survival Prediction App 🚢

This repository contains:
* `Train_Model.py` for training a **Titanic survival prediction model** using a Random Forest on the Titanic dataset.
* `app.py`, a **Streamlit web app** to predict survival chances for Titanic passengers using the trained model interactively.

## Features
- Train a robust survival model on Titanic data
- Predict survival probability based on passenger details
- Visualize prediction probabilities with Plotly
- Lightweight and easy to extend for ML beginners

## Setup Instructions

### 1️ Clone the repository

```bash
git clone <repository_link>
cd <repository_folder>
```

### 2️ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

---

### 3️ Install dependencies

```bash
pip install pandas scikit-learn joblib streamlit plotly
```

## Usage

###  1. Train the Model

Run the training script to download the Titanic dataset, train the Random Forest model, evaluate accuracy, and save the trained model as `best_survival_model.pkl`:

```bash
python Train_Model.py
```

Expected output:

```
Model Accuracy: 0.XXXX
Model saved as best_survival_model.pkl
```

###  2. Launch the Streamlit App

Run the Streamlit app locally:

```bash
streamlit run app.py
```

This will open the **Titanic Survival Prediction App** in your browser, where you can:

- Enter passenger details (age, class, fare, etc.)
- Get survival prediction and probabilities
- Visualize results using bar charts


## File Descriptions

* **Train\_Model.py** – Downloads Titanic data, preprocesses, trains the model, and saves it.
* **app.py** – Streamlit interface for user interaction and predictions.
* **best\_survival\_model.pkl** – The trained Random Forest model saved for inference.

## Screenshots

<img src="https://raw.githubusercontent.com/yourusername/yourrepo/main/screenshot.png" width="600"/>


## Future Improvements

* Hyperparameter tuning for improved accuracy.
* Adding feature explanations using SHAP or LIME.
* Dockerizing the app for smoother deployment.
