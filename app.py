# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:29:36 2021

@author: JASWANTH
"""
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10*1024*1024

def read_image(filename):
    img = image.load_img(filename , target_size = (32,32))
    img = image.img_to_array(img)
    img = img.reshape(1,32,32,3)
    return img

@app.route("/", methods = ['GET','POST'])
def home():
        return render_template('index.html')
   
@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        filename = file.filename
        file_path = os.path.join('user uploads' , filename)
        file.save(file_path)
        img = read_image(file_path)
        model1 = load_model('breast_cancer_classifier.h5')
        class_prediction = model1.predict_classes(img)
        if class_prediction[0] == 1:  #BENIGN
            result = "YOU ARE FREE FROM BREAST CANCER (BENIGN)"
        else:                         #MALIGNANT
            result = "Your tissues have BREAST CANCER (MALIGNANT), consult a doctor immediately, \nYou are a fighter you're going to beat this!"
    return render_template('predict.html', prediction = result)
        

if __name__ == "__main__":
    app.run(debug =True)
