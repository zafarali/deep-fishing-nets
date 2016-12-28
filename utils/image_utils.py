"""
	This file contains image utilities
	for dealing with files from the Nature Conservancy Kaggle Competition
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
from matplotlib.patches import Rectangle
from sklearn_theano.feature_extraction import OverfeatLocalizer
from sklearn.mixture import GMM

# These are the labels that seem to pick out most of the fish
CRAFTED_LABELS = ['electric ray, crampfish, numbfish, torpedo', 
	'goldfish, Carassius auratus',
	'anemone fish',
	'gar, garfish, garpike, billfish, Lepisosteus osseus']

# These are all the labels used
ALL_LABELS = ['grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
 'starfish, sea star',
 'electric ray, crampfish, numbfish, torpedo',
 'goldfish, Carassius auratus',
 'anemone fish',
 'lionfish',
 'puffer, pufferfish, blowfish, globefish',
 'gar, garfish, garpike, billfish, Lepisosteus osseus',
 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
 'crayfish, crawfish, crawdad, crawdaddy',
 'jellyfish',
 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 'tiger shark, Galeocerdo cuvieri',
 'hammerhead, hammerhead shark']

AUTO_RESIZE_DEFAULT = (720, 1028) # most common image size in the dataset

def read_image(src):
	return mpimg.imread(src) 

def imshow(img):
	plt.imshow(img)

def imresize(img, size=AUTO_RESIZE_DEFAULT):
	return misc.imresize(img, size, interp='nearest')

def convert_gmm_to_box(gmm, color='blue', alpha=0.6, points_only=True):
	"""
		Converts a mixture model approximating a location into a rectangular box
		@params:
			gmm: a sklear.mixture.GMM instance
			color[='blue']: the color to use for this box
			alpha[=0.6]: the transparency for this box
			points_only[=True]: returns only points for the box

		@returns:
			if points_only:
				upper_left_x, upper_left_y, width, height
			else:
				matplotlib.patches.Rectangle instance


		Taken from: http://sklearn-theano.github.io/auto_examples/plot_multiple_localization.html#example-plot-multiple-localization-py
	"""
	midpoint = gmm.means_
	std = 3*np.sqrt(gmm.covars_)
	width = std[:, 0]
	height = std[:, 1]
	upper_left_point_x = midpoint[:, 0] - width // 2
	upper_left_point_y = midpoint[:, 1] - height // 2

	if points_only:
		return upper_left_point_x, upper_left_point_y, width, height
	else:
		return Rectangle(upper_left_point, width, height, ec=color, fc=color, alpha=alpha)


class FishLocalizer(object):
	def __init__(self, labels=CRAFTED_LABELS):
		"""
			Detects fish from the OverFeat network Localizer
			Reference:
			P Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, Y. LeCun. 
			OverFeat: Integrated Recognition, Localization, and Detection 
			using Convolutional Networks
			International Conference on Learning Representations (ICLR 2014), April 2014.
		"""
		self.localizer = OverfeatLocalizer(match_strings=labels, top_n=2)

	def find(self, image, around=15):
		"""
			Find potential fish locations in an image
			@params
				image: the image to extract fish from
				around: the pixes around the detected fish position to extract from
			@returns
				list of images containing extracted fish
		"""
		points = self.localizer.predict(image)
		extracted_fish = []
		for p, point in enumerate(points):
			gmm = GMM()
			try:
				# use a GMM to find the fish positions
				gmm.fit(point)
				ulp_x, ulp_y, w, h = convert_gmm_to_box(gmm)
				ulp_x = int(ulp_x)-around if ulp_x > around else 0
				ulp_y = int(ulp_y)-around if ulp_y > around else 0
				w, h = int(w), int(h)
				if w == 0 or h == 0:
					continue

				# extract the fish
				extracted_fish.append(image[ulp_y: ulp_y+h+around, ulp_x: ulp_x+w+around, :])
			except Exception as e:
				# the gmm could not be fit, possibly due to not enough points
				# @TODO: check if the points are the reason for skipping here
				print str(e)
			#end try
		#endfor

		return extracted_fish


