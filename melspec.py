import numpy as np
import librosa as lb
import os

DATA_DIR = "openmic-2018/"
DATA_FILE = "openmic-2018.npz"
OPENMIC = np.load(os.path.join(DATA_DIR, DATA_FILE), allow_pickle=True)
Y_true, Y_mask, sample_key = OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
MEL = []
for i in range(sample_key.shape[0]):
	key = sample_key[i]
	key_prefix = key[:3]
	y, sr = lb.load(DATA_DIR + 'audio/' + key_prefix + '/' + key + '.ogg')
	S = lb.feature.melspectrogram(y=y, sr=sr)
	S_dB = lb.power_to_db(S, ref=0)
	MEL.append(S_dB[:,:430])
	if(i % 100 == 0):
		print(str(i) + ' samples are ready...')
	
MEL_S = np.asarray(MEL)
print('Mel has shape: ' + str(MEL_S.shape))
print('Writing file ' + DATA_DIR + 'openmic-mel.npz ...')
np.savez_compressed(DATA_DIR + 'openmic-mel.npz', X = MEL_S, Y_true=Y_true, Y_mask=Y_mask, sample_key=sample_key)
print('Done')