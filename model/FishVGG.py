"""
	Contains a classifier based on VGGNet
"""

from FishNet import FishNet
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import keras.layers as layers
import net_utils as utils
from keras.models import Model

class FishVGG(FishNet):
	def __init__(self, auto_resize=(90, 160)):
		"""
			Create a model for a fish classifier
		"""
		
		self.auto_resize = auto_resize 
		self.trained = False
		self.compiled = False
		self.datasets = None
		self.model_name = 'FishVGG'
		
		base_model = VGG16(include_top=False, 
			weights='imagenet', 
			input_tensor=layers.Input(auto_resize+(3,)))
			# input_tensor=layers.Activation(activation=utils.center_normalize, input_shape=auto_resize+(3,)))

		# freeze the VGG layers
		for layer in base_model.layers:
			layer.trainable = False
		
		x = base_model.output
		x = layers.Convolution2D(32, 2, 2, border_mode='same', activation='relu', name='outconv2d')(x)
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		# x = layers.Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='outconv2d')(x)
		# x = layers.MaxPooling2D(pool_size=(3, 3))(x)
		x = layers.Flatten(name='flatten')(x)
		x = layers.Dropout(0.5)(x)
		x = layers.Dense(256, activation='relu', name='fc1')(x)
		x = layers.Dropout(0.5)(x)
		x = layers.Dense(64, activation='relu', name='fc2')(x)
		x = layers.Dropout(0.5)(x)
		# x = layers.Dense(16, activation='relu', name='fc3')(x)
		x = layers.Dense(len(utils.ALL_FISH_CATEGORIES), activation='softmax', name='predictions')(x)

		self.model = Model(input=base_model.input, output=x)


	def init(self, lr=0.001, metrics=['categorical_crossentropy', 'categorical_accuracy'], **kwargs):
		"""
			Initializes the model using a stochastic gradient descent classifier
		"""

		sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

		super(FishVGG, self).init(loss='categorical_crossentropy', optimizer=sgd, metrics=metrics, **kwargs)


	