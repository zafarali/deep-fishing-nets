import os, sys

def add_module_paths():
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	

import keras.layers as layers
from keras import backend as K

ALL_FISH_CATEGORIES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def add_conv_max_layers(model, n_filters=32, filter_size=(5,5), activation='relu', pool_size=(2,2)):
	model.add(layers.Convolution2D(n_filters, filter_size[0], filter_size[1], border_mode='same', activation=activation))
	model.add(layers.MaxPooling2D(pool_size=pool_size))
	return model

def fish_out(model):
	model.add(layers.Dense(len(io.ALL_FISH_CATEGORIES)))
	model.add(layers.Activation('softmax'))
	return model

def center_normalize(x):
	x = np.float(x) / 255.0
	return (x - K.mean(x)) / K.std(x)

W, H, CH = (90,160,3)
def fish_in(model, activation=center_normalize, input_shape=(W, H, CH)):
	model.add(layers.Activation(activation=activation, input_shape=input_shape))
	return model
