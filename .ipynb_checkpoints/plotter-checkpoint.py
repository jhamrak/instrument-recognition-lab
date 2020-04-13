# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from datetime import datetime
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow, xlim, ylim, title
import os


def plot_history(history, instrument):
    plt.figure(figsize=(9,4))
    plt.title(instrument)
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper left')
    os.mkdir(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    plt.savefig(datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '/' + instrument +'.png')
