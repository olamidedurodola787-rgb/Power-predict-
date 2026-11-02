
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load("powerpredict_model.pkl")
feature_names = joblib.load("powerpredict_features.pkl")

st.set_page_config(page_title="PowerPredict", page_icon="⚡")

st.title("⚡ PowerPredict")
st.subheader("Electricity Consumption Prediction")

st.write(
    "Predict household electricity consumption "
    "using Machine Learning."
)

st.info("Enter your previous 24 hourly consumption readings.")

readings = st.text_area(
    "Previous 24 readings (kW)",
    placeholder="Example: 1.2, 1.5, 1.3, ..."
)

prediction_time = st.time_input(
    "Select prediction time",
    value=datetime.now().time()
)

prediction_date = st.date_input(
    "Select prediction date"
)

if st.button("Predict Consumption"):

    try:
        values = [float(x.strip()) for x in readings.split(",")]

        if len(values) != 24:
            st.error("Please enter exactly 24 readings.")
        else:
            next_time = datetime.combine(
                prediction_date, prediction_time
            )

            features = {}

            for i in range(1, 25):
                features[f"lag_{i}"] = [values[i-1]]

            features["Hour"] = [next_time.hour]
            features["Day"] = [next_time.weekday()]

            input_data = pd.DataFrame(features)
            input_data = input_data[feature_names]

            result = model.predict(input_data)[0]

            st.success("Prediction completed!")
            st.metric(
                "Predicted Electricity Consumption",
                f"{result:.3f} kW"
            )

    except ValueError:
        st.error("Enter valid numbers separated by commas.")
