import streamlit as st
import cv2
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from streamlit_image_coordinates import streamlit_image_coordinates
from io import StringIO


st.header('Color Recognition App 🌎')
#if st.button('Balloons?'):
#    st.balloons()

bytes_data = None
img_file_buffer = st.camera_input('Snap a picture')
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	bytes_data = img_file_buffer.getvalue()

elif uploaded_file is not None:
	bytes_data = uploaded_file.getvalue()

if bytes_data is None:
	st.stop()


dataset = pd.read_csv('color_names.csv')
X = dataset.drop('colour', axis=1)
y = dataset['colour']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
st.text(f'({img.shape[1]}x{img.shape[0]})')
st.header('colour detected')

cv2.imwrite('ImageCaptured.jpg', img)
value = streamlit_image_coordinates('ImageCaptured.jpg', height=900, key="local",)
#st.write(value)

HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = HSVImg[value['y'], value['x']]
#hsv = HSVImg[0, 0]

HSVvalue = ''
h = int(hsv[0]) / 180 * 360
s = int(hsv[1]) / 255 * 100
v = int(hsv[2]) / 255 *100
HSVvalue = str(h) + ',' + str(s) + ',' + str(v)

colour_prediction = knn.predict([[h,s,v]])

## for matching colours
matching_colours_dataset = pd.read_csv('matching_colours.csv')
matching_colours_list = matching_colours_dataset['complement'].dropna()
matching_colours_list = matching_colours_list.values.tolist()
find = 'purple'
a = -1
        
for colours in matching_colours_list:
	if colour_prediction[0] in colours:
	#if Colour_name in colours:
        	a = matching_colours_list.index(colours)

#if ():
#	matching_colours.append(matching_colours_dataset[0])

st.text('hsv value: ' + HSVvalue)
st.text('colour name: ' + colour_prediction[0])
#st.text('colour name2: ' + Colour_name)
st.text('Suggested matching colour for ' + colour_prediction[0] + ':')
st.text('classic match: ' + ' & '.join(matching_colours_dataset['basic'].values.tolist()))
st.text('complement match: ' + matching_colours_list[a])
