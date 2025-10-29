import streamlit as st
import joblib
import pickle
import pandas as pd 
import numpy as np

kmeans=joblib.load('model_kmeans.pkl')
scaler=joblib.load('scaler.pkl')
st.title('Customer Segmentation App')
st.subheader('Developed by Anas Athar')
st.write('Enter The Customer Details...')
age=st.number_input('Age',min_value=18,max_value=100,value=28)
income=st.number_input('Income',min_value=10,max_value=20000,value=5000)
total_spending=st.number_input('Total Spending (sum of purchases)',min_value=0,max_value=5000,value=1000)
number_web_purchases=st.number_input('Num of Web Purchases',min_value=0,max_value=100 ,value=10)
num_store_purchases=st.number_input('Num of Store Purchases',min_value=0,max_value=100,value=10)
num_web_visits_month=st.number_input('Num of Web Visits per Months',min_value=0,max_value=50,value=3)
recency=st.number_input('Recency (day since last purchase)',min_value=0,max_value=362,value=30)

input_data=pd.DataFrame({
    'Age':[age],
    'Income':[income],
    'Total_Spending':[total_spending],
    'NumWebPurchases':[number_web_purchases],
    'NumStorePurchases':[num_store_purchases],
    'NumWebVisitsMonth':[num_web_visits_month],
    'Recency':[recency]
})
cluster_names = {
    0: 'High Value Customers',
    1: 'Frequent Shoppers',
    2: 'Occasional Buyers',
    3: 'New Customers',
    4: 'Online Enthusiasts',
    5: 'Low Engagement'
}
input_scaler=scaler.transform(input_data)

if st.button('Predict:'):
    cluster=kmeans.predict(input_scaler)[0] #index like value 1 if not use git got a array[2]
    cluster_name=cluster_names.get(cluster,f'Unknown Segment')
    st.success(f'Predicted Segment: {cluster_name} (Cluster {cluster})')
