from fastapi import FastAPI, Form, File, UploadFile
from starlette.responses import HTMLResponse
from skimage.io import imread
from skimage.transform import resize
import pickle
import numpy as np
from pathlib import Path
import math
import decimal
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img

app = FastAPI()

@app.get('/index',response_class=HTMLResponse)
def main_page():
	return'''
		<b> Choose one method</b>
		<p><a href="http://127.0.0.1:8000/MLclassifier"> ML model classifier</a></p>
		<p><a href="http://127.0.0.1:8000/DLclassifier"> DL model classifier</a></p>
		'''

#ML classifier part
@app.get('/MLclassifier', response_class=HTMLResponse)
def take_input():
	return '''
		<form action="/MLEyePrediction" method="post"> 
		<b>Enter Eye Image path</b>
		<input type="text" value="path" name="text1"/>
		<input type="submit" value="SuBmIt"/>
		</form>
		'''
		
def MLimg_process(txt):
	flat_data = []
	print("yoklink")
	img_arr = imread(txt) 
	print("tomako")
	img_resized = resize(img_arr,(50,50,3))
	flat_data.append(img_resized.flatten())
	return flat_data
	
@app.post('/MLEyePrediction')

def predict(text1:str=Form()):
	print(text1)
	print("yoklines")
	text1 = text1.strip('"')
	processed_img = MLimg_process(Path(text1))
	loaded_model = pickle.load(open("C:/Users/Com/Downloads/Compressed/Eyes Data Set/logisticmodel", 'rb'))
	predictions = loaded_model.predict(processed_img)
	prob = loaded_model.predict_proba(processed_img)
	#log_prob = loaded_model.predict_log_proba(processed_img)
	print(prob)
	#print(log_prob)
	print(predictions)
	threshold = 10**(-8)
	#gender = np.argmax(predictions)
	print("0:",prob[0][0], ",1:",prob[0][1])
	if (prob[0][0]<threshold	or prob[0][1]<threshold):
		error = "Could not identify image!!"
		return error
	else: 
		if predictions == 0:
			eye = "Female Eye" 
		elif predictions == 1:
			eye = "Male Eye"
		return eye
	print("Complete")
	
	

#DL classifier part
@app.get('/DLclassifier', response_class=HTMLResponse)
def take_input():
	return '''
		<form action="/DLEyePrediction" method="post"> 
		<b>Enter Eye Image path</b>
		<input type="text" value="path" name="text1"/>
		<input type="submit" value="sUbMiT"/>
		</form>
		'''
		
def DLimg_process(txt):
	
	img_src = load_img(txt,target_size=(224,224))
	print("tomako")
	#For "preprocess_input" we need to convert image using tf.cast
	x = tf.cast(img_src, tf.float32)
	print("yoklink")
	DLprocessed_img = preprocess_input(x)
	print(DLprocessed_img.shape)
	return DLprocessed_img

	
@app.post('/DLEyePrediction')

def predict(text1:str=Form()):
	print(text1)
	print("tunutunu")
	text1 = text1.strip('"')
	processed_img = DLimg_process(Path(text1))
	loaded_model = load_model("C:/Users/Com/Downloads/Compressed/Eyes Data Set/tensorEye.h5")
	processed_img2= np.reshape(processed_img,(1,224,224,3))
	print("Input image shape is:",processed_img2.shape)
	predictions = loaded_model.predict(processed_img2)
	#prob = loaded_model.predict_classes(processed_img2)
	#log_prob = loaded_model.predict_log_proba(processed_img)
	#print(prob)
	print(predictions)
	
	if predictions[0][0]>= predictions[0][1]:
			eye = "Female Eye" 
	elif predictions[0][0]<= predictions[0][1]:
			eye = "Male Eye"
	return eye
	
	
	'''
	print(predictions)
	threshold = 10**(-8)
	#gender = np.argmax(predictions)
	print("0:",prob[0][0], ",1:",prob[0][1])
	if (prob[0][0]<threshold	or prob[0][1]<threshold):
		error = "Could not identify image!!"
		return error
	else: 
		if predictions == 0:
			eye = "Female Eye" 
		elif predictions == 1:
			eye = "Male Eye"
		return eye
	print("Complete")
    '''