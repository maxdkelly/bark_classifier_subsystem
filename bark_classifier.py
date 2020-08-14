import os
import sys


import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from keras.models import load_model

if len(sys.argv) != 2:
	print("Invalid Number of Arguments!")

else:  
	IMG_HEIGHT = 224
	IMG_WIDTH = 224
	feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
	file_name = 'tmp_bark_img.jpg'

	feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_HEIGHT,IMG_WIDTH,3))


	feature_extractor_layer.trainable = False

	model = tf.keras.Sequential([
    	feature_extractor_layer,
    	keras.layers.Dense(3)
	])

	
	model.load_weights('./weights/bark_weights')
	data_dir = "bark_segments/"
	path = data_dir + sys.argv[1]

	print(path)

	classes = ['high','low','moderate']
	image = tf.keras.preprocessing.image.load_img(path ,target_size = (IMG_WIDTH, IMG_HEIGHT,3))
	input_arr = keras.preprocessing.image.img_to_array(image, dtype = tf.keras.backend.floatx() )

	input_arr = np.kron(input_arr,1/255)

	input_arr = np.array([input_arr])  # Convert single image to a batch.
	predictions = model.predict(input_arr)

	predict_arr = predictions.tolist()[0]

	print(classes[predict_arr.index(max(predict_arr))])
	
