import librosa as lb


def get_transformed_array(X_old):
	X = X_old
	shape = X.shape
	X = X.astype('float16')
	X = X.reshape(shape[0],1, shape[1], shape[2])
	X = lb.util.normalize(X)
	return X