import argparse

def DeepLearning(DATA_FILE, LEARNING_RATE, THRESHOLD, EPOCHS):

	import librosa as lb
	import librosa.display
	import scipy
	import json
	import numpy as np
	import sklearn
	from sklearn.metrics import classification_report
	from sklearn.model_selection import train_test_split
	import os
	import keras
	from keras.utils import np_utils
	from keras import layers
	from keras import models
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
	from keras.preprocessing.image import ImageDataGenerator
	from model_builder import build_example
	import matplotlib.pyplot as plt
	import plotter
	import utils
	import model_builder
	from contextlib import redirect_stdout
	from datetime import datetime

	# CONSTANTS
	DATA_DIR = "openmic-2018/"
	CLASS_MAP = "class-map.json"
	CATEGORY_COUNT = 8
	INPUT_SHAPE = (1,10,128)

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
	print('<< Loading data: ' + DATA_DIR + '/' + DATA_FILE)

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

	# TRAIN AND EVALUATE


	print('-' * 52)
	print('>> Tranining and evaluation')
	dir = utils.create_dir()
	mymodels = dict()
	raw_model = model_builder.build_small(INPUT_SHAPE)
	with open(dir + "/log.txt", "a") as f:
		with redirect_stdout(f):
			raw_model.summary()
	for instrument in INSTRUMENTS:

		# Map the instrument name to its column number
		inst_num = INSTRUMENTS[instrument]

		# Step 1: sub-sample the data
		# First, we need to select down to the data for which we have annotations
		# This is what the mask arrays are for
		# Here, we're using the Y_mask_train array to slice out only the training examples
		# for which we have annotations for the given class
		# Again, we slice the labels to the annotated examples
		# We thresold the label likelihoods at 0.5 to get binary labels

		# TRAIN
		train_inst = Y_mask_train[:, inst_num]
		X_train_inst = utils.get_transformed_array(X_train[train_inst])
		Y_true_train_inst = Y_true_train[train_inst, inst_num] >= THRESHOLD

		# TEST
		test_inst = Y_mask_test[:, inst_num]
		X_test_inst = utils.get_transformed_array(X_test[test_inst])
		Y_true_test_inst = Y_true_test[test_inst, inst_num] >= THRESHOLD

		# VALIDATION
		val_inst = Y_mask_val[:, inst_num]
		X_val_inst = utils.get_transformed_array(X_val[val_inst])
		Y_true_val_inst = Y_true_val[val_inst, inst_num] >= THRESHOLD

		model = raw_model
		model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr= LEARNING_RATE), metrics = ['accuracy'])
		
		print('-' * 52)
		print('\t' + instrument.upper())
		print('>> Tranining model for [' + instrument + ']')
		history = model.fit(X_train_inst,Y_true_train_inst , epochs=EPOCHS, batch_size=64, validation_data=(X_val_inst,Y_true_val_inst))
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
	args = parser.parse_args()
	
	if args.data.upper() == 'VGG':
		data = 'openmic-2018.npz'
	elif args.data.upper() == 'MELSPEC':
		data = 'openmic-mel.npz'
	elif args.data.upper() == 'MFCC':
		data = 'openmic-mfcc.npz'
	else:
		raise ValueError('[DATA] param error')
	
	if args.mode.upper() == 'DL':
		DeepLearning(data, float(args.lr), float(args.threshold), int(args.epochs))
	elif args.mode.upper() == 'ML':
		DeepLearning() #TODO ML 
	else:
		raise ValueError('[MODE] param error')
	