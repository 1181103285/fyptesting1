import streamlit as st

import cv2
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


st.header('Hello 🌎!')
if st.button('Balloons?'):
    st.balloons()
    
    
bytes_data = None
img_file_buffer = st.camera_input('Snap a picture')
if img_file_buffer is not None:
	bytes_data = img_file_buffer.getvalue()
	
if bytes_data is None:
	st.stop()
	
dataset = pd.read_csv('color_names.csv')
dataset = dataset.drop(columns=['Hex (24 bit)', 'Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)'])
#dataset
X = dataset.drop('Name', axis=1)
y = dataset['Name']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=7)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

