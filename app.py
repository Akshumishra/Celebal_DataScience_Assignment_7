# app.py

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.title("Titanic Survival Prediction App")
st.write("This application predicts the likelihood of survival for a passenger on the Titanic based on various features.")

# Load trained model
try:
    model = joblib.load('best_survival_model.pkl')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'best_survival_model.pkl' not found. Please ensure the model is trained and saved in the app directory.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

if model:
    st.header("Enter Passenger Information:")

    # Collect user input
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
    parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.number_input("Fare", min_value=0.0, value=50.0)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

    # Encode categorical variables
    sex_encoded = 1 if sex == 'male' else 0
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    embarked_encoded = embarked_mapping[embarked]

    # Create DataFrame aligned with training data columns
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success("Passenger is likely to survive!")
        else:
            st.error("Passenger is likely to not survive.")

        st.subheader("Prediction Probabilities:")
        survival_proba = prediction_proba[0][1]
        not_survival_proba = prediction_proba[0][0]
        st.write(f"Probability of Not Surviving: `{not_survival_proba:.2f}`")
        st.write(f"Probability of Surviving: `{survival_proba:.2f}`")

        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(name='Probability', x=['Not Survived', 'Survived'], y=[not_survival_proba, survival_proba],
                   marker_color=['crimson', 'green'])
        ])
        fig.update_layout(title_text='Survival Probability', yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
