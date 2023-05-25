import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import r2_score
# r2 only available in tf-nightly
#https://www.tensorflow.org/api_docs/python/tf/keras/metrics/R2Score

import matplotlib.pyplot as plt
from itertools import product
from math import floor
import time

import tensorflow as tf
import tensorflow_federated as tff
#import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.callbacks import CSVLogger



def prep_fed_test(X_test: pd.DataFrame, y_test: pd.DataFrame):
  """Converts Testdata to Tensor Object.

  See https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#preprocessing_the_input_data


  Args:
      X_test (pd.DataFrame): Test features.
      y_test (pd.DataFrame): Test target.

  Returns:
      A `Tensor` based on `X_test`, `y_test`.
  """
  return tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(np.expand_dims(X_test, axis=0)), 
    tf.convert_to_tensor(np.expand_dims(y_test, axis=0))
    )) 
  
help(prep_fed_test)
