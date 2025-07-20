# train_titanic_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Titanic dataset
try:
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

if df is not None:
    # Select relevant columns
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    df.dropna(inplace=True)

    # Encode categorical features
    df['Sex'] = df['Sex'].astype('category').cat.codes  # female=0, male=1
    df['Embarked'] = df['Embarked'].astype('category').cat.codes  # C=0, Q=1, S=2

    # Features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save the trained model
    joblib.dump(model, 'best_survival_model.pkl')
    print("Model saved as best_survival_model.pkl")

else:
    print("Could not proceed with model training due to data loading error.")
