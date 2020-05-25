import os
import librosa as lb
from datetime import datetime
import numpy as np

def undersample(X,Y,inst_coords,inst_num, threshold):
	X_inst = X[inst_coords]
	Y_inst = Y[inst_coords, inst_num] >= threshold
	count = np.sum(Y_inst)
	i = 0
	undersampled_coords = []
	for i in range(len(Y_inst)) :
		if Y_inst[i] == 1:
			undersampled_coords.append(i)
		elif count > 0:
			undersampled_coords.append(i)
			count -= 1
	return X_inst[undersampled_coords], Y_inst[undersampled_coords]

def get_instrument_arrays(X, Y, mask, inst_num, threshold):
	inst_coords = mask[:, inst_num]
	X, Y = undersample(X,Y,inst_coords,inst_num, threshold)
	X = get_transformed_array(X)
	return X, Y
	
def get_instrument_arrays_ml(X, Y, mask, inst_num, threshold):
	inst_coords = mask[:, inst_num]
	X, Y = undersample(X,Y,inst_coords,inst_num, threshold)
	X = get_normalized_array(X)
	return X, Y


def get_transformed_array(X_old):
	X = X_old
	shape = X.shape
	X = X.astype('float16')
	X = X.reshape(shape[0],1, shape[1], shape[2])
	X = lb.util.normalize(X)
	return X
	
def get_normalized_array(X_old):
	X = X_old
	X = X.astype('float16')
	X = lb.util.normalize(X)
	return X
	
def create_dir(mode, data):
	dir_name = "logs/" + datetime.now().strftime("%m%d%H%M%S") + "-" + mode + "-" + data
	os.mkdir(dir_name)
	return dir_name