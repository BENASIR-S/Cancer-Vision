
from __future__ import division, print_function
# coding=utf-8
import re
import sys
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model
from keras import backend
from tensorflow.keras import backend

from skimage.transform import resize

# Flask utils
from flask import Flask, app, request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Load your trained model
model = load_model(r"breastcancer.h5",compile = False)
#print('Model loaded. Check http://127.0.0.1:5000/')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

from logging import FileHandler,WARNING
# Define a flask app
app = Flask(__name__,template_folder='template')
app.config['UPLOAD_FOLDER'] = "breastcancerdataset"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('/bcancer.html')#,name=name)


@app.route('/predict', methods=["POST","GET"])
def predict():
    if request.method == "POST":
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)

        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #  with graph.as_default():
        preds = np.argmax(model.predict(x))
        
        if preds[0][0] == 0:
            text = "The tumor is benign.. Need not worry!"
        else:
            text = "It is a malignant tumor... Please Consult Doctor"
        text = text

        # ImageNet Decode

        return text


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
