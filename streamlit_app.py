import streamlit as st
import cv2
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.header('Color Recognition App 🌎')
if st.button('Balloons?'):
    st.balloons()
	
#cap = cv2.VideoCapture(0)
#(ret, frame) = cap.read()
#prediction = 'n.a.'

#cap.release()

## for matching colours
matching_colours_dataset = pd.read_csv('matching_colours.csv', header=None)
matching_colours_list = matching_colours_dataset.to_numpy()
st.text('testing: ' + str(matching_colours_list))

#st.text('testing: ' + matching_colours_list[0])
matching_colours = ''
#if ():
#	matching_colours.append(matching_colours_dataset[0])


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

## for matching colours
matching_colours_dataset = pd.read_csv('matching_colours.csv', header=None)
matching_colours_list = matching_colours_dataset.to_numpy()
st.text('testing: ' + matching_colours_list[0][0] + matching_colours_list[0][1])

#st.text('testing: ' + matching_colours_list[0])
matching_colours = ''
#if ():
#	matching_colours.append(matching_colours_dataset[0])

st.text('hsv value: ' + HSVvalue)
st.text('colour name: ' + colour_prediction[0])
st.text('Suggested matching colour: ' + colour_prediction[0])
