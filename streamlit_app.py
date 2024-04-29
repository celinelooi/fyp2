import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('trained_model.joblib')

model = load_model()

# Set up the Streamlit app
st.title('Football Match Predictor')
st.markdown("""
This app predicts the outcome of a football match. Please select the match details below and click 'Predict Outcome' to see the prediction.
""")

# User inputs
date = st.date_input("Match Date", datetime.today())
time = st.time_input("Match Time")
venue = st.selectbox('Venue', ['Home', 'Away', 'Neutral'])
opponent = st.selectbox('Opponent', ['Team A', 'Team B', 'Team C', 'Team D'])

# Convert inputs to appropriate format
venue_code = int(venue == 'Home')
opp_code = {'Team A': 0, 'Team B': 1, 'Team C': 2, 'Team D': 3}[opponent]
hour = time.hour
day_code = date.weekday()

# Prediction button
if st.button('Predict Outcome'):
    with st.spinner('Calculating...'):
        try:
            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'venue_code': [venue_code],
                'opp_code': [opp_code],
                'hour': [hour],
                'day_code': [day_code]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            outcome = 'Win' if prediction[0] == 1 else 'Loss'
            st.success(f'The predicted outcome is: {outcome}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")
