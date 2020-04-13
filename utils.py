import os
import librosa as lb
from datetime import datetime


def get_transformed_array(X_old):
	X = X_old
	shape = X.shape
	X = X.astype('float16')
	X = X.reshape(shape[0],1, shape[1], shape[2])
	X = lb.util.normalize(X)
	return X
	
def create_dir():
	dir_name = "logs/" + datetime.now().strftime("%Y%m%d%H%M%S")
	os.mkdir(dir_name)
	return dir_name