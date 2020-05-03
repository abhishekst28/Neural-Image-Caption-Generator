
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from gtts import gTTS 

from pickle import load
from numpy import argmax


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


model1= VGG16()
model1.layers.pop()
model1 = Model(inputs=model1.inputs, outputs=model1.layers[-1].output)

MODEL_PATH = 'models/model_cap.h5'
model2 = load_model(MODEL_PATH)
#model2._make_predict_function() 

def extract_features(filename):

	#model = model.load_weights('models/vgg16_weights.h5')
	#model= VGG16(weights='models/vgg16_weights.h5')
	#model.layers.pop()
	#model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model1.predict(image, verbose=0)
	return feature

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_desc(tokenizer, photo, max_length):

	in_text = 'startseq'

	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model2.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text


app = Flask(__name__)

print('Model loaded. Start serving...')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


class ReturnValue(object):
  __slots__ = ["y0", "y1"]
  def __init__(self, y0, y1):
     self.y0 = y0
     self.y1 = y1
     

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print('Uploaded file : '+ str(file_path))
        tokenizer = load(open('models/tokenizer.pkl', 'rb'))
        
        max_length = 34
        
        photo = extract_features(file_path)
        result = generate_desc(tokenizer, photo, max_length)
        result = result.replace("startseq", "")
        result = result.replace("endseq", "")
        mytext = result
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False) 
        audio_name=str(file_path)
        audio_name=audio_name.replace("uploads\\","")
        audio_name=audio_name.replace("jpg","mp3")
        audio_name=audio_name.replace("JPG","mp3")
        audio_name=audio_name.replace("png","mp3")
        audio_name=audio_name.replace("PNG","mp3")
        print(audio_name)
        audio_name="audio_files/"+audio_name
        myobj.save(audio_name)

        #return result
        print("This is abs path :"+os.path.dirname(os.path.realpath(__file__)))
        #file_name="start C:/Users/toshn/Desktop/mits/"+audio_name #add your system's path here
        abc=os.path.dirname(os.path.realpath(__file__))
        abc=abc.replace("\\","/")
        #abc=abc.replace("static/j","")
        file_name=abc+"/"+audio_name
        print("this is file name of mp3 being played"+file_name)
        os.system(file_name)

        print(type('{} {}'.format(result, audio_name)))
        print('{} {}'.format(result, audio_name))
        return  '{} {} {}'.format(result,"delimits", audio_name)
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

