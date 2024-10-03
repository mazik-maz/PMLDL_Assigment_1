import streamlit as st
import requests

# Streamlit UI elements
st.title("Titanic Survival Predictor")
st.write("Enter the passenger details:")

# Input fields
pclass = st.selectbox("Passenger Class (1, 2, 3)", [1, 2, 3])
sex = st.selectbox("Sex (0 = male, 1 = female)", [0, 1])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.01)
embarked = st.selectbox("Port of Embarkation (0 = S, 1 = C, 2 = Q)", [0, 1, 2])

# Button to make a prediction
if st.button("Predict"):
    # Prepare input data
    data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }

    # Send a request to FastAPI for prediction
    response = requests.post("http://api:8000/predict/", json=data)
    
    if response.status_code == 200:
        result = response.json()["survived"]
        st.success(f"Prediction: {'Survived' if result == 1 else 'Did Not Survive'}")
    else:
        st.error("Error in prediction request")