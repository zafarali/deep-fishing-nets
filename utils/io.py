"""
	This file contains input-output utilities
	for dealing with files from the Nature Conservancy Kaggle Competition
"""
from glob import glob
from random import sample as sampler

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
	return sampler(all_img_urls, sample) if sample > 0 else all_img_urls

