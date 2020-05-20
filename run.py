import argparse
import os
import scipy
import json
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import plotter
import utils
from contextlib import redirect_stdout

DATA_DIR = "openmic-2018/"
DATASET = ''
CLASS_MAP = "class-map.json"

def read_data(DATA_FILE):
	
	# LOAD DATA
	print('-' * 52)
	print('>> Loading data: ' + DATA_DIR + DATA_FILE)
	OPENMIC = np.load(os.path.join(DATA_DIR, DATA_FILE), allow_pickle=True)
	X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
	print('Keys: ' + str(list(OPENMIC.keys())))
	print('X shape: ' + str(X.shape))
	print('Y_true shape: ' + str(Y_true.shape))
	print('Y_mask shape: ' + str(Y_mask.shape))
	print('sample_key shape: ' + str(sample_key.shape))
	print('<< Loading data: ' + DATA_DIR + DATA_FILE)

	# LOAD LABELS

	print('-' * 52)
	print('>> Loading labels: ' + CLASS_MAP)
	with open(os.path.join(DATA_DIR, CLASS_MAP), 'r') as f:
		INSTRUMENTS = json.load(f)
	print('Instrument labels loaded:')
	for instrument in INSTRUMENTS:
		print(str(INSTRUMENTS[instrument]) + ' : ' + instrument)
	print('<< Loading labels: ' + CLASS_MAP)

	# SPLIT DATA (TRAIN - TEST - VAL)

	print('-' * 52)
	print('>> Splitting data TRAIN/TEST')
	X_train, X_test, Y_true_train, Y_true_test, Y_mask_train, Y_mask_test = train_test_split(X, Y_true, Y_mask)
	print('<< Splitting data TRAIN/TEST')
	print('>> Splitting data TEST/VAL')
	X_val, X_test, Y_true_val, Y_true_test, Y_mask_val, Y_mask_test = train_test_split(X_test, Y_true_test, Y_mask_test, test_size=0.5)
	print('# Train: {}, # Val: {}, # Test: {}'.format(len(X_train), len(X_test), len(X_val)))
	print('<< Splitting data TEST/VAL')
	return X_train, Y_true_train, Y_mask_train, X_test, Y_true_test, Y_mask_test, X_val, Y_true_val, Y_mask_val, INSTRUMENTS

def MachineLearning(DATA_FILE, THRESHOLD):

	from sklearn.ensemble import RandomForestClassifier

	X_train, Y_true_train, Y_mask_train, X_test, Y_true_test, Y_mask_test, X_val, Y_true_val, Y_mask_val, INSTRUMENTS = read_data(DATA_FILE)
	print('-' * 52)
	print('>> Tranining and evaluation')
	dir = utils.create_dir('ML',DATASET)
	for instrument in INSTRUMENTS:

		# Map the instrument name to its column number
		inst_num = INSTRUMENTS[instrument]

		# TRAIN
		X_train_inst, Y_true_train_inst = utils.get_instrument_arrays(X_train, Y_true_train, Y_mask_train, inst_num, THRESHOLD)
		print('train shapes')
		print(str(X_train_inst.shape))
		print(str(Y_true_train_inst.shape))

		# TEST
		X_test_inst, Y_true_test_inst = utils.get_instrument_arrays(X_test, Y_true_test, Y_mask_test, inst_num, THRESHOLD)
		print('test shapes')
		print(str(X_test_inst.shape))
		print(str(Y_true_test_inst.shape))

		# VALIDATION
		X_val_inst, Y_true_val_inst = utils.get_instrument_arrays(X_val, Y_true_val, Y_mask_val, inst_num, THRESHOLD)
		print('val shapes')
		print(str(X_val_inst.shape))
		print(str(Y_true_val_inst.shape))
		X_train_inst_sklearn = np.mean(X_train_inst, axis=1)
		X_test_inst_sklearn = np.mean(X_test_inst, axis=1)

		clf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
		
		print('-' * 52)
		print('\t' + instrument.upper())
		print('>> Tranining model for [' + instrument + ']')
		clf.fit(X_train_inst_sklearn, Y_true_train_inst)
		print('<< Tranining model for [' + instrument + ']')
		
		print('>> Logging to file for [' + instrument + ']')
		
		Y_pred_train = clf.predict(X_train_inst_sklearn)
		Y_pred_test = clf.predict(X_test_inst_sklearn)
		Y_pred_train_bool = Y_pred_train > THRESHOLD #(should be lower ???)
		Y_pred_test_bool = Y_pred_test > THRESHOLD #(should be lower ???)
		
		with open(dir + "/log.txt", "a") as f:
			with redirect_stdout(f):
				print('-' * 52)
				print(instrument)
				print('\tTRAIN')
				print(classification_report(Y_true_train_inst, Y_pred_train_bool))
				print('\tTEST')
				print(classification_report(Y_true_test_inst, Y_pred_test_bool))
		print('<< Logging to file for [' + instrument + ']')
		
	print('<< Tranining and evaluation')

def DeepLearning(DATA_FILE, LEARNING_RATE, THRESHOLD, EPOCHS, INST_COUNT):

	import keras
	from keras.utils import np_utils
	from keras import layers
	from keras import models
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
	from keras.preprocessing.image import ImageDataGenerator
	import model_builder
	from keras.callbacks import EarlyStopping

	es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto')

	X_train, Y_true_train, Y_mask_train, X_test, Y_true_test, Y_mask_test, X_val, Y_true_val, Y_mask_val, INSTRUMENTS = read_data(DATA_FILE)

	# TRAIN AND EVALUATE


	print('-' * 52)
	print('>> Tranining and evaluation')
	dir = utils.create_dir('DL',DATASET)
	mymodels = dict()
	raw_model = model_builder.build_small((1, X_train.shape[1], X_train.shape[2]))
	with open(dir + "/log.txt", "a") as f:
		with redirect_stdout(f):
			print('Runtime params: ')
			print('Dataset: ' + DATASET)
			print('Learning rate: ' + str(LEARNING_RATE))
			print('Threshold: ' + str(THRESHOLD))
			print('Number of epochs: ' + str(EPOCHS))
			print('-' * 52)
			raw_model.summary()
	for instrument in INSTRUMENTS:

		# Map the instrument name to its column number
		inst_num = INSTRUMENTS[instrument]
		if inst_num > 3 and inst_num < (3 + INST_COUNT):
			# TRAIN
			X_train_inst, Y_true_train_inst = utils.get_instrument_arrays(X_train, Y_true_train, Y_mask_train, inst_num, THRESHOLD)
			print('train shapes')
			print(str(X_train_inst.shape))
			print(str(Y_true_train_inst.shape))

			# TEST
			X_test_inst, Y_true_test_inst = utils.get_instrument_arrays(X_test, Y_true_test, Y_mask_test, inst_num, THRESHOLD)
			print('test shapes')
			print(str(X_test_inst.shape))
			print(str(Y_true_test_inst.shape))

			# VALIDATION
			X_val_inst, Y_true_val_inst = utils.get_instrument_arrays(X_val, Y_true_val, Y_mask_val, inst_num, THRESHOLD)
			print('val shapes')
			print(str(X_val_inst.shape))
			print(str(Y_true_val_inst.shape))

			model = raw_model
			model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr= LEARNING_RATE), metrics = ['accuracy'])
			
			print('-' * 52)
			print('\t' + instrument.upper())
			print('>> Tranining model for [' + instrument + ']')
			history = model.fit(X_train_inst,Y_true_train_inst , epochs=EPOCHS, batch_size=64, callbacks=[es], validation_data=(X_val_inst,Y_true_val_inst))
			print('<< Tranining model for [' + instrument + ']')
			
			print('>> Evaluating model for [' + instrument + ']')
			loss, acc = model.evaluate(X_test_inst, Y_true_test_inst)
			print('Test loss: {}'.format(loss))
			print('Test accuracy: {:.2%}'.format(acc))
			print('<< Evaluating model for [' + instrument + ']')
			
			print('>> Logging to file for [' + instrument + ']')
			plotter.plot_history(history, instrument, dir)
			
			Y_pred_train = model.predict(X_train_inst)
			Y_pred_test = model.predict(X_test_inst)
			Y_pred_train_bool = Y_pred_train > THRESHOLD #(should be lower ???)
			Y_pred_test_bool = Y_pred_test > THRESHOLD #(should be lower ???)
			
			with open(dir + "/log.txt", "a") as f:
				with redirect_stdout(f):
					print('-' * 52)
					print(instrument)
					print('\tTRAIN')
					print(classification_report(Y_true_train_inst, Y_pred_train_bool))
					print('\tTEST')
					print(classification_report(Y_true_test_inst, Y_pred_test_bool))
			print('<< Logging to file for [' + instrument + ']')
					
			mymodels[instrument] = model
		
	print('<< Tranining and evaluation')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='DL', required=False)
	parser.add_argument('--data', default='VGG', required=False)
	parser.add_argument('--lr', default='0.0001', required=False)
	parser.add_argument('--threshold', default='0.5', required=False)
	parser.add_argument('--epochs', default='10', required=False)
	parser.add_argument('--instruments', default='5', required=False)
	args = parser.parse_args()
	
	if args.data.upper() == 'VGG':
		DATASET = 'VGG'
		data = 'openmic-2018.npz'
	elif args.data.upper() == 'MELSPEC':
		DATASET = 'MEL'
		data = 'openmic-mel.npz'
	elif args.data.upper() == 'MFCC':
		DATASET = 'MFCC'
		data = 'openmic-mfcc.npz'
	else:
		raise ValueError('[DATA] param error')
	
	if args.mode.upper() == 'DL':
		DeepLearning(data, float(args.lr), float(args.threshold), int(args.epochs), int(args.instruments))
	elif args.mode.upper() == 'ML':
		MachineLearning(data,  float(args.threshold))
	else:
		raise ValueError('[MODE] param error')
	