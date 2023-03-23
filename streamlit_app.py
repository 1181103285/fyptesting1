import streamlit as st
import cv2
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.header('Color Recognition App 🌎')
if st.button('Balloons?'):
    st.balloons()

bytes_data = None
img_file_buffer = st.camera_input('Snap a picture')
if img_file_buffer is not None:
	bytes_data = img_file_buffer.getvalue()
	
if bytes_data is None:
	st.stop()

#dataset = pd.read_csv('color_names.csv')
#dataset = dataset.drop(columns=['Hex (24 bit)', 'Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)'])
#X = dataset.drop('Name', axis=1)
#y = dataset['Name']
#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=7)
#knn = KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train, y_train)

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


####
hues = {0: 'red', 15: 'red-orange', 30: 'orange', 45: 'orange-yellow', 60: 'yellow', 
	75: 'yellow-chartreuse', 90: 'chartreuse', 105: 'chatreuse-green', 120: 'green', 
	135: 'green-jade', 150: 'jade', 165: 'jade-cyan', 180: 'cyan', 
	195: 'cyan-azure', 210: 'azure', 225: 'azure-blue', 240: 'blue', 
	255: 'blue-violet', 270: 'violet', 285: 'violet-magenta', 300: 'magenta', 
	315: 'magenta-rose', 330: 'rose', 345: 'rose-red', 360: 'red'}

Hue_for_dict = min(hues, key=lambda x:abs(x-h))
Hue_name = hues[Hue_for_dict]

Saturation_dict = {10: 'grayish', 35: 'desaturated', 60: 'saturated', 85: 'very saturated'}
Saturation_for_dict = min(Saturation_dict, key=lambda x:abs(x-s))
Saturation_name = Saturation_dict[Saturation_for_dict]

Lightness_dict = {10: 'and very dark',35: 'dark', 60: 'light', 85: 'and very light'}
Lightness_for_dict = min(Lightness_dict, key=lambda x:abs(x-v))
Lightness_name = Lightness_dict[Lightness_for_dict]

Colour_name = ''
if s <= 5:
	if v < 2:
		Colour_name = 'black'
	elif v < 45:
		Colour_name = 'dark gray'
	elif v < 70:
		Colour_name = 'gray'
	elif v < 95:
		Colour_name = 'light gray'
	else:
		Colour_name = 'white'
else:
	Colour_name = "%s %s %s" %(Saturation_name, Lightness_name, Hue_name)

####
#colour_prediction = knn.predict([[h,s,v]])

## for matching colours
matching_colours_dataset = pd.read_csv('matching_colours.csv')
matching_colours_list = matching_colours_dataset['complement'].dropna()
matching_colours_list = matching_colours_list.values.tolist()
find = 'purple'
a = -1
        
for colours in matching_colours_list:
	#if colour_prediction[0] in colours:
	if Colour_name in colours:
        	a = matching_colours_list.index(colours)

#if ():
#	matching_colours.append(matching_colours_dataset[0])

st.text('hsv value: ' + HSVvalue)
#st.text('colour name: ' + colour_prediction[0])
st.text('colour name2: ' + Colour_name)
st.text('Suggested matching colour for ' + Colour_name + ':')
st.text('classic match: ' + ' & '.join(matching_colours_dataset['basic'].values.tolist()))
st.text('complement match: ' + matching_colours_list[a])
