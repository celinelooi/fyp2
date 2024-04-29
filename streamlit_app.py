import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime

# Load the trained model
model = load('trained_model.joblib')

# Title and description
st.title('Football Match Predictor')
st.write('Enter the details of the match to predict the outcome.')

# Input fields for match details
team = st.selectbox('Team', options=['Liverpool', 'Manchester City', 'Other teams'])  # Update with all options
opponent = st.selectbox('Opponent', options=['Team A', 'Team B', 'Other teams'])  # Update with all options
venue = st.selectbox('Venue', options=['Home', 'Away'])
date = st.date_input('Date', min_value=datetime.now())
time = st.time_input('Kick-off Time')

# Convert inputs to model input format
venue_code = 1 if venue == 'Home' else 0
opp_code = 1  # Update based on actual encoding
hour = time.hour
day_code = date.weekday()  # Monday is 0 and Sunday is 6

# Prediction button
if st.button('Predict Result'):
    # Prepare the features as expected by the model
    features = pd.DataFrame({
        'venue_code': [venue_code],
        'opp_code': [opp_code],
        'hour': [hour],
        'day_code': [day_code]
    })
    
    # Predict
    prediction = model.predict(features)
    result = 'Win' if prediction[0] == 1 else 'Lose'
    
    # Display the prediction
    st.write(f'The prediction for {team} against {opponent} is: {result}')

# Optional: display model accuracy or other statistics
