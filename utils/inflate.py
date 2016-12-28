from scipy.ndimage.interpolation import rotate as imrotate
from image_utils import imresize
import numpy as np
import io

def generate_extra_data(X=None, Y=None, n=3000, seed=None, folder=io.DATA_FOLDER+'train', file_name=None, rotate=True, noise=True):
	print "Creating {} new images".format(n)
	
	if X is None and Y is None:
		# automatically resizes images to (720, 1028)
		X, _, _ = io.create_test_train_split(samples=-1, auto_resize=True, folder=folder)
		X, Y = io.dataset_dict_to_array(X)
	
	if seed:
		np.random.seed(seed)
	
	positions = np.random.randint(0, X.shape[0]-1, n)
	transformed = X[positions,:]
	targets = Y[positions, :]
	final = np.zeros_like(transformed)
	final_resized_190_320 = np.zeros((n, 190, 320, 3))
	final_resized_244_244 = np.zeros_like((n, 244, 244, 3))
	final_resized_48_48 = np.zeros_like((n, 48, 48, 3))

	# slight pertubations in the angle of the image
	angles = np.random.uniform(-20,+20, n)
	for i in range(n):
		temp = transformed[i,:]

		if rotate:
			temp = imrotate(temp, angles[i], reshape=False)
		if noise:
			rand_noise = np.random.randint(0, 255, temp.shape)
			keepers = np.random.binomial(1,0.95,size=temp.shape)
			temp = temp*keepers + rand_noise*(1-keepers)
			
		final[i,:] = temp
		final_resized_190_320[i, :] = imresize(temp, size=(190, 320))
		final_resized_244_244[i, :] = imresize(temp, size=(244, 244))
		final_resized_48_48[i, :] = imresize(temp, size=(48, 48))

		if i % 1000 == 0:
			print "Created {} images.".format(i)

	if file_name:
		if not file_name.startswith("./data/"):
			file_name = folder + file_name
		np.save(file_name+'.npy', final)
		np.save(file_name+'_190_320.npy', final_resized_190_320)
		np.save(file_name+'_244_244.npy', final_resized_244_244)
		np.save(file_name+'_48_48.npy', final_resized_48_48)
		np.save(file_name + "_targets.npy", targets)
	return final, targets
