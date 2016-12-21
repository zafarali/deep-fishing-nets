"""
	This file contains input-output utilities
	for dealing with files from the Nature Conservancy Kaggle Competition
"""
from glob import glob
from random import sample as sampler
from image_utils import read_image, imresize

DATA_FOLDER = '../data/'
ALL_FISH_CATEGORIES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def get_image_paths(fish, sample=-1, folder='train'):
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

	FISH_PATH = DATA_FOLDER+folder+'/'+fish+'/*.jpg'
	all_img_urls = glob(FISH_PATH)
	try:
		return sampler(all_img_urls, sample) if sample > 0 else all_img_urls
	except ValueError as e:
		return all_img_urls

def create_test_train_split(categories=ALL_FISH_CATEGORIES, split=0.8, samples=100, auto_resize=False):
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
		paths = get_image_paths(fish, sample=100)
		if type(auto_resize) is tuple and len(auto_resize) == 2:
			# resize to specified value
			imgs = [ imresize(read_image(path), size=auto_resize) for path in paths ]
		if type(auto_resize) is bool and auto_resize:
			# automatically resizes to image_utils.AUTO_RESIZE_DEFAULT
			imgs = [ imresize(read_image(path)) for path in paths ]
		else:
			# no resize
			imgs = [ read_image(path) for path in paths ]
		X[fish] = imgs
		X_train[fish] = imgs[ :int(split * len(imgs)) ]
		X_val[fish] = imgs[ int(split * len(imgs)): ]
	return X, X_train, X_val

