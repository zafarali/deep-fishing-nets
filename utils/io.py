"""
	This file contains input-output utilities
	for dealing with files from the Nature Conservancy Kaggle Competition
"""
from glob import glob
import numpy as np
import pandas as pd
from random import sample as sampler
from image_utils import read_image, imresize
from keras.utils import np_utils

DATA_FOLDER = '../data/'
ALL_FISH_CATEGORIES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def get_image_paths(fish, sample=-1, folder=DATA_FOLDER+'train'):
	"""
		Returns image paths for a class of fish
		@params:
			fish: the fish category to return
			sample [=-1]: the number of files to return
				(if negative, returns all)
			folder [=train]: loads images from a 
				specified subdirectory
	"""
	assert fish in ALL_FISH_CATEGORIES, 'unknown fish requested.'

	FISH_PATH = folder+'/'+fish+'/*.jpg'
	print 'loading from:' + FISH_PATH
	all_img_urls = glob(FISH_PATH)
	try:
		return sampler(all_img_urls, sample) if sample > 0 else all_img_urls
	except ValueError as e:
		return all_img_urls

def load_imgs_from_paths(paths, auto_resize=True):
	
	# shuffle paths so that when we do test/train splits
	# we get shuffled distributions
	paths = sampler(paths, len(paths))

	if type(auto_resize) is tuple and len(auto_resize) == 2:
		# resize to specified value
		imgs = [ imresize(read_image(path), size=auto_resize) for path in paths ]
	elif type(auto_resize) is bool and auto_resize:
		# automatically resizes to image_utils.AUTO_RESIZE_DEFAULT
		imgs = [ imresize(read_image(path)) for path in paths ]
	else:
		# no resize
		imgs = [ read_image(path) for path in paths ]
	return imgs


def create_test_train_split(folder=DATA_FOLDER+'train', categories=ALL_FISH_CATEGORIES, split=0.8, samples=100, auto_resize=False):
	"""
		Loads a dataset into memory pre-split into train and validation
		@params:
			categories: the list of fish labels to load (by default all)
			split[=0.8]: the % of the dataset to use as train
			samples[=100]: the number of images to load per category
			auto_resize[=False]: if given a tuple will automatically resize images to this size.
		@returns
			X, X_train, X_val datasets keyed by the categories.
	"""
	X, X_train, X_val = {}, {}, {}

	for fish in categories:
		paths = get_image_paths(fish, sample=samples, folder=folder)

		imgs = load_imgs_from_paths(paths, auto_resize=auto_resize)

		X[fish] = imgs
		X_train[fish] = imgs[ :int(split * len(imgs)) ]
		X_val[fish] = imgs[ int(split * len(imgs)): ]
	return X, X_train, X_val

def dataset_dict_to_array(X):
	"""
		Turns a dictionary of {'label': np.array of size n*m } into 
		X, Y
		@params:
			X: the dict
		@returns:
			X: the input data
			Y: the labels one-hot-encoded
	"""
	x = []
	Y = np.zeros(sum([ len(xi) for xi in X.values() ]))
	for i, label in enumerate(sorted(X.keys())):
		start = len(x) # the starting length of x
		x.extend(X[label])
		end = len(x) # the ending length of x

		Y[start:end] = i # fill in between start and end 

	return np.asarray(x, dtype=np.uint8), np_utils.to_categorical(Y)

def load_test_data(test_folder=DATA_FOLDER+'test_stg1', auto_resize=True):
	"""	
		Loads the test data
		@params:
			test_folder[='test_stg1']: the folder in which the files are stored
		@returns:
			test_set: np.array of size N*auto_resize
			paths: the list of file names
	"""
	paths = glob(test_folder+'/*.jpg')
	imgs = load_imgs_from_paths(paths, auto_resize=auto_resize)
	
	test_set = np.array(imgs)
	paths = [path.split('/')[-1] for path in paths]

	return test_set, paths

def save_predictions(test_filenames, predictions, out_filename='submission.csv', location='./'):
	"""
		Saves predictions from a model
		@params:
			test_filenames: the name of the test files
			predictions: the predictions for each file as an nparray
			out_filename[='submission.csv']: the filename to save the submission
			location[='./']: the folder to save the submission
		@returns:
			None
	"""
	submission = pd.DataFrame(predictions, columns=ALL_FISH_CATEGORIES)
	submission.insert(0, 'image', test_filenames)
	submission.to_csv(location+out_filename, index=False)
	print 'Predictions saved to '+location+out_filename
