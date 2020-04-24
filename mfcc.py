import numpy as np
import librosa as lb
import os

DATA_DIR = "openmic-2018/"
DATA_FILE = "openmic-2018.npz"
OPENMIC = np.load(os.path.join(DATA_DIR, DATA_FILE), allow_pickle=True)
Y_true, Y_mask, sample_key = OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
MFCC = []
for i in range(sample_key.shape[0]):
	key = sample_key[i]
	key_prefix = key[:3]
	y, sr = lb.load(DATA_DIR + 'audio/' + key_prefix + '/' + key + '.ogg')
	S = lb.feature.mfcc(y=y, sr=sr)
	MFCC.append(S[:,:430])
	if(i % 100 == 0):
		print(str(i) + ' samples are ready...')

MFCC_S = np.asarray(MFCC)
print('MFCC has shape: ' + str(MFCC_S.shape))

np.savez_compressed(DATA_DIR + 'openmic-mfcc.npz', X = MFCC_S, Y_true=Y_true, Y_mask=Y_mask, sample_key=sample_key)
