import streamlit as st
import numpy as np
import pandas as pd
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
  y_raw = df.Class
  y_raw
