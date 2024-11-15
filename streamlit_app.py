import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.title('👨‍⚕️ Machine Learning App')
st.info('')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/GenTpham/RandomForest/refs/heads/main/cancer.csv')
  df

  st.write('**X**')
  df = df.drop('ID', axis = 1)
  X_raw = df.drop('Class', axis = 1)
  X_raw

  st.write('**y**')
  y = df.Class
  y

with st.sidebar:
  st.header('Input fearures')
  #Clump,UnifSize,UnifShape,MargAdh,SingEpiSize,BareNuc,BlandChrom,NormNuc,Mit
  Clump = st.slider('Clump', 0, 20, 10)
  UnifSize = st.slider('UnifSize', 0, 20, 10)
  UnifShape = st.slider('UnifShape', 0, 20, 10)
  MargAdh = st.slider('MargAdh', 0, 20, 10)
  SingEpiSize = st.slider('SingEpiSize', 0,20,10)
  BareNuc = st.slider('BareNuc', 0,20,10)
  BlandChrom = st.slider('BlandChrom', 0, 20, 10)
  NormNuc = st.slider('NormNuc', 0,20,10)
  Mit = st.slider('Mit', 0,20,10)

  data = {'Clump': Clump,
          'UnifSize': UnifSize,
          'UnifShape': UnifShape,
          'MargAdh': MargAdh,
          'SingEpiSize': SingEpiSize,
          'BareNuc': BareNuc,
          'BlandChrom': BlandChrom,
          'NormNuc': NormNuc,
          'Mit': Mit}
  input_df = pd.DataFrame(data, index =[0])
  input_cancer = pd.concat([input_df, X_raw], axis = 0)

with st.expander('Input features'):
  st.write('**Input cancer**')
  input_df
  st.write('**Combined cancer data**')
  input_cancer

X = input_cancer[1:]
input_row = input_cancer[:1]
std = StandardScaler()
X = std.fit_transform(X)
clf = SVC()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['benign', 'malignant']
df_prediction_proba.rename(columns = {'benign': 2,
                                     'malignant': 4})
