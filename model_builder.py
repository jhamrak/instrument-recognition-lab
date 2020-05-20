import keras
from keras.utils import np_utils
from keras import layers
from keras import models
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def build_vgg(in_shape):
	model = models.Sequential()
	model.name = 'VGG'
	model.add(Conv2D(filters=64 , kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first", input_shape=in_shape))
	model.add(Conv2D(filters=64 , kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(Dropout(0.2))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(Dropout(0.2))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(Dropout(0.2))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(layers.Flatten())
	model.add(layers.Dense(units=4096, activation='relu'))
	model.add(layers.Dense(units=2048, activation='relu'))
	model.add(layers.Dense(units=1, activation='sigmoid'))
	return model
	
def build_small(in_shape):
	model = models.Sequential()
	model.name = 'VGG small'
	model.add(Conv2D(filters=64 , kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first", input_shape=in_shape))
	model.add(Conv2D(filters=64 , kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(Dropout(0.2))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", data_format="channels_first"))
	model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
	model.add(Dropout(0.2))
	model.add(layers.Flatten())
	model.add(layers.Dense(units=128, activation='relu'))
	model.add(layers.Dense(units=1, activation='sigmoid'))
	return model