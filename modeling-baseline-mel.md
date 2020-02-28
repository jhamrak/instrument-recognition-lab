---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# OpenMIC-2018 baseline model tutorial

This notebook demonstrates how to replicate a simplified version of the baseline modeling experiment in [(Humphrey, Durand, and McFee, 2018)](http://ismir2018.ircam.fr/doc/pdfs/203_Paper.pdf).

First, make sure you [download the dataset](https://zenodo.org/record/1432913#.W6dPeJNKjOR)!

We'll load in the pre-computed [VGGish features](https://github.com/tensorflow/models/tree/master/research/audioset) and labels, and fit a [RandomForest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model for each of the 20 instrument classes using the pre-defined train-test splits provided in the repository.

We'll then evaluate the models we fit, and show how to apply them to new audio signals.

This notebook is not meant to demonstrate state-of-the-art performance on instrument recognition.  Rather, we hope that it can serve as a starting point for building your own instrument detectors without too much effort!

```python
# These dependencies are necessary for loading the data
import json
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Be sure to set this after downloading the dataset!
DATA_ROOT = 'openmic-2018/'

if not os.path.exists(DATA_ROOT):
    raise ValueError('Did you forget to set `DATA_ROOT`?')
```

## Loading the data

The openmic data is provided in a python-friendly format as `openmic-2018.npz`.

You can load it as follows:

```python
OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'),allow_pickle=True)
```

```python
# What's included?
print(list(OPENMIC.keys()))
```

### What's included in the data?

- `X`: 20000 * 10 * 128 array of VGGish features
    - First index (0..19999) corresponds to the sample key
    - Second index (0..9) corresponds to the time within the clip (each time slice is 960 ms long)
    - Third index (0..127) corresponds to the VGGish features at each point in the 10sec clip
    - Example `X[40, 8]` is the 128-dimensional feature vector for the 9th time slice in the 41st example
- `Y_true`: 20000 * 20 array of *true* label probabilities
    - First index corresponds to sample key, as above
    - Second index corresponds to the label class (accordion, ..., voice)
    - Example: `Y[40, 4]` indicates the confidence that example #41 contains the 5th instrument
- `Y_mask`: 20000 * 20 binary mask values
    - First index corresponds to sample key
    - Second index corresponds to the label class
    - Example: `Y[40, 4]` indicates whether or not we have observations for the 5th instrument for example #41
- `sample_key`: 20000 array of sample key strings
    - Example: `sample_key[40]` is the sample key for example #41

```python
# It will be easier to use if we make direct variable names for everything
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
```

```python
X.shape
```

```python
# Features for the 9th time slice of 81st example
X[80, 8]
```

```python
Y_true[40]
```

```python
Y_mask[40]
```

```python
sample_key.shape
```

```python
sample_key[40]
```

### Load the class map

For convenience, we provide a simple JSON object that maps class indices to names.


```python
with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
    class_map = json.load(f)
```

```python
class_map
```

---


## Loading the train-test splits

OpenMIC-2018 comes with a pre-defined train-test split.  Great care was taken to ensure that this split is approximately balanced and artists are not represented in both sides of the split, so please use it!

This is done by sample key, not row number, so you will need to go through the `sample_key` array to slice the data.

```python
# Let's split the data into the training and test set
# We use squeeze=True here to return a single array for each, rather than a full DataFrame

split_train = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_train.csv'), 
                          header=None, squeeze=True)
split_test = pd.read_csv(os.path.join(DATA_ROOT, 'partitions/split01_test.csv'), 
                         header=None, squeeze=True)
```

```python
# These two tables contain the sample keys for training and testing examples
# Let's see the keys for the first five training example
split_train.head(5)
```

```python
# How many train and test examples do we have?  About 75%/25%
print('# Train: {},  # Test: {}'.format(len(split_train), len(split_test)))
```


These sample key maps are easier to use as sets, so let's make them sets!

```python
train_set = set(split_train)
test_set = set(split_test)
```

### Split the data

Now that we have the sample keys for the training and testing examples, we need to partition the data arrays (`X`, `Y_true`, `Y_mask`).

This is a little delicate to get right.

```python
# These loops go through all sample keys, and save their row numbers
# to either idx_train or idx_test
#
# This will be useful in the next step for slicing the array data
idx_train, idx_test = [], []

for idx, n in enumerate(sample_key):
    if n in train_set:
        idx_train.append(idx)
    elif n in test_set:
        idx_test.append(idx)
    else:
        # This should never happen, but better safe than sorry.
        raise RuntimeError('Unknown sample key={}! Abort!'.format(sample_key[n]))
        
# Finally, cast the idx_* arrays to numpy structures
idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)
```

```python
# Finally, we use the split indices to partition the features, labels, and masks
X_train = X[idx_train]
X_test = X[idx_test]

Y_true_train = Y_true[idx_train]
Y_true_test = Y_true[idx_test]

Y_mask_train = Y_mask[idx_train]
Y_mask_test = Y_mask[idx_test]
```

```python
# Print out the sliced shapes as a sanity check
print(X_train.shape)
print(X_test.shape)
```

---
# Fit the models

Now, we'll iterate over all the instrument classes, and fit a separate `RandomForest` model for each one.

For each instrument, the steps are as follows:

1. Find the subset of training (and testing) data that have been annotated for the current instrument
2. Simplify the features to have one observation point per clip, instead of one point per time slice within each clip
3. Initialize a classifier
4. Fit the classifier to the training data
5. Evaluate the classifier on the test data and print a report


```python
# This dictionary will include the classifiers for each model
models = dict()

# We'll iterate over all istrument classes, and fit a model for each one
# After training, we'll print a classification report for each instrument
for instrument in class_map:
    
    # Map the instrument name to its column number
    inst_num = class_map[instrument]
        
    # Step 1: sub-sample the data
    
    # First, we need to select down to the data for which we have annotations
    # This is what the mask arrays are for
    train_inst = Y_mask_train[:, inst_num]
    test_inst = Y_mask_test[:, inst_num]
    
    # Here, we're using the Y_mask_train array to slice out only the training examples
    # for which we have annotations for the given class
    X_train_inst = X_train[train_inst]
    
    # Step 3: simplify the data by averaging over time
    
    # Let's arrange the data for a sklearn Random Forest model 
    # Instead of having time-varying features, we'll summarize each track by its mean feature vector over time
    X_train_inst_sklearn = np.mean(X_train_inst, axis=1)
    
    # Again, we slice the labels to the annotated examples
    # We thresold the label likelihoods at 0.5 to get binary labels
    Y_true_train_inst = Y_true_train[train_inst, inst_num] >= 0.5

    
    # Repeat the above slicing and dicing but for the test set
    X_test_inst = X_test[test_inst]
    X_test_inst_sklearn = np.mean(X_test_inst, axis=1)
    Y_true_test_inst = Y_true_test[test_inst, inst_num] >= 0.5

    # Step 3.
    # Initialize a new classifier
    clf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
    
    # Step 4.
    clf.fit(X_train_inst_sklearn, Y_true_train_inst)

    # Step 5.
    # Finally, we'll evaluate the model on both train and test
    Y_pred_train = clf.predict(X_train_inst_sklearn)
    Y_pred_test = clf.predict(X_test_inst_sklearn)
    
    print('-' * 52)
    print(instrument)
    print('\tTRAIN')
    print(classification_report(Y_true_train_inst, Y_pred_train))
    print(Y_true_train_inst[3])
    print(Y_pred_train[3])
    print('\tTEST')
    print(classification_report(Y_true_test_inst, Y_pred_test))
    
    print(Y_true_test_inst.shape)
    print(Y_pred_test.shape)
    
    # Store the classifier in our dictionary
    models[instrument] = clf
```

---


# Applying the model to new data

In this section, we'll take the models trained above and apply them to audio signals, stored as OGG Vorbis files.

```python
# We need soundfile to load audio data
import soundfile as sf

# And the openmic-vggish preprocessor
import openmic.vggish

# For audio playback
from IPython.display import Audio
```

```python
# We include a test ogg file in the openmic repository, which we can use here.
audio, rate = sf.read(os.path.join(DATA_ROOT, 'audio/000/000046_3840.ogg'))

time_points, features = openmic.vggish.waveform_to_features(audio, rate)
```

```python
# The time_points array marks the starting time of each observation
time_points
```

```python
# The features array includes the vggish feature observations
features.shape
```

```python
# Let's listen to the example
Audio(data=audio.T, rate=rate)
```

```python
# finally, apply the classifier

# Average over time to one observation, but keep the number of dimensions the same
# The test clip is 10sec long, so this is the same process as in the training step
# However, you could also apply the classifier to each frame independently to get time-varying predictions
feature_mean = np.mean(features, axis=0, keepdims=True)

for instrument in models:
    
    clf = models[instrument]
    
    print('P[{:18s}=1] = {:.3f}'.format(instrument, clf.predict_proba(feature_mean)[0,1]))
```

# Wrapping up

So the predictions here are definitely not perfect, but they're a good start!

Some things you might want to try out:

1. Instead of averaging features over time, apply the classifiers to each time-step to get a time-varying instrument detector.
2. Play with the parameters of the `RandomForest` model, changing the depth and number of estimators.
3. Run the trained model on your own favorite songs!
4. Train a different model, maybe using different features!
5. Make use of label uncertainties or unlabeled data when training!
