"""
	Contains an abstract FishNet Class
"""
import os, sys

def add_module_paths():
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	
from utils import io
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

class FishNet(object):
	def __init__(self, model, auto_resize=(90, 160)):
		"""
			Create a model for a fish classifier
		"""
		self.model = model
		self.auto_resize = auto_resize 
		self.trained = False
		self.compiled = False
		self.datasets = None
		self.model_name = 'FishNet'

	def init(self, loss, optimizer, metrics=['categorical_crossentropy', 'categorical_accuracy'], **kwargs):
		"""
			Initializes the model
		"""
		self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
		print self.model.summary()
		self.compiled = True

	def load_training_images(self, split=0.8, samples=100, folder='./data/train/'):
		X, X_train, X_val = io.create_test_train_split(folder=folder, auto_resize=self.auto_resize, split=split, samples=samples)

		x, y = io.dataset_dict_to_array(X)
		x_train, y_train = io.dataset_dict_to_array(X_train)
		x_val, y_val = io.dataset_dict_to_array(X_val)
		self.datasets = {
			'full': (x, y), 
			'training': (x_train, y_train),
			'validation': (x_val, y_val)
			}

		return self.datasets['full'], self.datasets['training'], self.datasets['validation']


	def train(self, x=None, y=None, epochs=1, callbacks=[], batch_size=32, **kwargs):
		"""
			Fits the model using x and y for epochs
		"""
		if x is None and y is None:
			if self.datasets is None:
				raise ValueError('No dataset to train with')
			else:
				x, y = self.datasets['training']

		callbacks = callbacks + [ EarlyStopping(patience=2, verbose=1),
								ModelCheckpoint('./'+self.model_name+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)]
		self.model.fit(x, y, nb_epoch=epochs, validation_split=0.1, batch_size=batch_size, shuffle=True, verbose=1, callbacks=callbacks, **kwargs)
		self.trained = True

	def evaluate(self, x=None, y=None):
		"""
			Evaluates the FishNet model
			if X and Y are not specified, will search for self.datasets to evaluate
		"""
		if x is None and y is None:
			if self.datasets is None:
				raise ValueError('No dataset to evaluate with')
			else:
				x, y = self.datasets['validation']

		preds = self.model.predict(x, verbose=1)
		print "Validation Log Loss: "+str(log_loss(y, preds))

	def test(self, file_name='submission.csv', folder='./data/test_stg1'):
		"""
			Runs model on a test set to create a submission for kaggle
		"""
		x_test, paths = io.load_test_data(auto_resize=self.auto_resize, test_folder=folder)
		preds = self.model.predict(x_test, verbose=1)
		io.save_predictions(paths, preds, out_filename=file_name)

	@staticmethod
	def load(self, file_name, auto_resize=(90,160)):
		model = load_model(file_name)
		fnn = FishNet(model, auto_resize=auto_resize)
		fnn.trained = True
		fnn.compiled = True
		return fnn


	