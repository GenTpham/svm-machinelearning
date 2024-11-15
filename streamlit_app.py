import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.title('👨‍⚕️ Machine Learning App')
st.info('')

# Load and display data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/GenTpham/RandomForest/refs/heads/main/cancer.csv')
    st.dataframe(df)
    
    st.write('**X**')
    df = df.drop('ID', axis=1)
    X_raw = df.drop('Class', axis=1)
    st.dataframe(X_raw)
    
    st.write('**y**')
    y = df.Class
    st.dataframe(y)

# Sidebar inputs
with st.sidebar:
    st.header('Input features')
    features = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 
                'BareNuc', 'BlandChrom', 'NormNuc', 'Mit']
    
    input_data = {}
    for feature in features:
        input_data[feature] = st.slider(feature, 0, 20, 10)
    
    input_df = pd.DataFrame([input_data])

# Display input features
with st.expander('Input features'):
    st.write('**Input data**')
    st.dataframe(input_df)

# Prepare data for modeling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
input_scaled = scaler.transform(input_df)

# Train model and make predictions
clf = SVC(probability=True)
clf.fit(X_scaled, y)
prediction = clf.predict(input_scaled)
prediction_proba = clf.predict_proba(input_scaled)

# Display results
st.subheader('Prediction Results')
df_prediction_proba = pd.DataFrame(prediction_proba, 
                                 columns=['benign', 'malignant'])
st.dataframe(df_prediction_proba)

st.write(f'**Predicted Class:** {prediction[0]}')
