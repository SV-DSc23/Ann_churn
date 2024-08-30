import streamlit as st
import numpy as np
import pandas as pd
#import scikit-learn
#from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
#import tensorflow as tf
import pickle

#Loading the model:
model = load_model("model.h5")

#Loading the encoders:
with open ("le_gender.pkl", 'rb') as file:
    le_gender = pickle.load(file)

with open ("ohe_geo.pkl", "rb") as file:
    ohe_geo = pickle.load(file)

#Loading the scaler:
with open ("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Streamlit app
st.title("Customer Churn Prediction")
#Useing input
CreditScore = st.number_input("CreditScore")
Geography = st.selectbox("Geography", ohe_geo.categories_[0])
Gender = st.selectbox("Gender", le_gender.classes_)
Age = st.slider("Age", 18,92)
Tenure = st.slider("Tenure", 0,11)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("NumOfProducts", 1,5)
HasCrCard = st.selectbox("HasCrCard", [0,1])
IsActiveMember = st.selectbox ("IsActiveMember", [0,1])
EstimatedSalary = st.number_input("EstimatedSalary")

#Preparing input
input_data = pd.DataFrame({
    "CreditScore" : [CreditScore],
    "Geography" : [Geography],
    "Gender" : [le_gender.transform([Gender])[0]],
    "Age" : [Age],
    "Tenure" : [Tenure],
    "Balance" : [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary],
})

#ohe geography
geo_ohencoded = ohe_geo.transform([[Geography]]).toarray()
geo_ohencoded_df = pd.DataFrame(geo_ohencoded, columns = ohe_geo.get_feature_names_out(["Geography"]))

#Concatinating encoded Geography data
input_data = pd.concat([input_data.drop("Geography", axis=1), geo_ohencoded_df], axis=1)

#Scaling input_data
scaled_input = scaler.transform(input_data)

#Predict churn
prediction = model.predict(scaled_input)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write('The customer is likely to CHURN.')
else:
    st.write('The customer is NOT likely to churn.')
