import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title('CUSTOMER CHURN PREDICTION')

model=load_model('model.h5')

with open('geo_encoder.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

credit_score = st.number_input("CreditScore",value=502)
geography = st.selectbox("Geography",geo_encoder.categories_[0],index=0)
gender = st.selectbox("Gender", ["Female", "Male"],index=0)
age = st.slider("Age",18,90,value=42)
tenure = st.slider("Tenure", 0,10,value=8)
balance = st.number_input("Balance",value=159660.80)
num_of_products = st.slider("NumOfProducts",1,4,value=3)
has_cr_card = st.slider("HasCrCard", 0,1,value=1)
is_active_member = st.slider("IsActiveMember", 0,1,value=0)
estimated_salary = st.number_input("EstimatedSalary",value=113931.57)


# 2. Put them directly into a DataFrame
input_df= pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])



geo_data=geo_encoder.transform([input_df['Geography']])
geo_df=pd.DataFrame(geo_data.toarray(),columns=['France','Germany','Spain'])


input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])

input_dff=pd.concat([input_df.drop(['Geography'],axis=1),geo_df],axis=1)
input_dat=scaler.transform(input_dff)

predict_pb=model.predict(input_dat)

st.write(f"Churn Rate:{predict_pb[0][0]}")

if(predict_pb[0][0]>=0.5):
    st.write("Customer is likely to exited")
else:
    st.write("Customer is not likely to exited")