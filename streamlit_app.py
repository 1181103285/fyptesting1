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


img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
st.text(f'({img.shape[1]}x{img.shape[0]})')
st.header('colour detected')

HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = HSVImg[0, 0]

HSVvalue = ''
h = int(hsv[0])
s = int(hsv[1]) / 255
v = int(hsv[2]) / 255
HSVvalue = str(h) + ',' + str(s) + ',' + str(v)

colour_prediction = knn.predict([[h,s,v]])

st.text('hsv value: ' + HSVvalue)
st.text('colour name: ' + colour_prediction[0])
