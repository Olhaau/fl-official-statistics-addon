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


def load_df(paths):
    """ Loads data from a path to a csv-file.
    
    :param df_locs: possible locations of a CSV file
    :type df_locs: str or list of str
    :output: Ingested Data.
    :rtype: pandas.DataFrame 
    """
    df = pd.DataFrame()

    if isinstance(paths, str): paths = [paths]
    
    for path in paths:
        try:
            df = pd.read_csv(path, index_col = 0)
            print("loaded data from {}".format(path))
            if len(df) != 0: break
        except Exception as ex:
            print("{} in ".format(type(ex).__name__), path)

    return df

def prep_fed_train(X_train, y_train):
    """
    See https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#preprocessing_the_input_data
    """

    return tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(X_train), 
        tf.convert_to_tensor(y_train)
        ))

def prep_fed_test(X_test, y_test):
   return tf.data.Dataset.from_tensor_slices((
       tf.convert_to_tensor(np.expand_dims(X_test, axis=0)), 
       tf.convert_to_tensor(np.expand_dims(y_test, axis=0))
       )) 

def create_keras_model(
    nfeatures = 9,
    units = [40, 40, 20], 
    activations = ['relu'] * 3, 
    compile = True,
    loss = tf.losses.mae,
    optimizer = tf.optimizers.legacy.Adam(learning_rate = .05),
    metrics = ["mae", 'mean_squared_error', r2_score], 
    run_eagerly = True
    ):
  
  """Construct a fully connected neural network and compile it.
  
  Parameters
  ------------
  nfeatures: int, optional
    Number of input features. Default is 9.
  units: list of int, optional
    List of number of units of the hidden dense layers. The length of ``units`` defines the number of hidden layers. Default are 3 layers with 40, 40 an 20 units, respectively.
  activations: list of str, optional
    List of activation functions used in the hidden layers.
  loss: str, optional
    Used loss function for compiling.
  optimizer: keras.optimizers, optional
    Used optimizer for compiling.
  metrics: list of str or sklearn.metrics
    List of metrics for compiling.
  run_eagerly: bool
    Parameter for compiling

  Return
  ------------
    model: keras.engine.sequential.Sequential
      Keras sequential fully connected neural network. Already compiled.
  """
  
  # construct model
  model = Sequential()
  model.add(InputLayer(input_shape = [nfeatures]))
  for ind in range(len(units)):
    model.add(Dense(
      units = units[ind], 
      activation = activations[ind]
      ))
  model.add(Dense(1))
  
  # compile model
  if compile:
    model.compile(
      loss = loss,
      optimizer = optimizer,
      metrics = metrics,
      run_eagerly = run_eagerly
    )

  return model

def model_fn(
    keras_creator,
    loss = tf.keras.losses.MeanAbsoluteError()
    #,metrics = [tf.keras.metrics.MeanAbsoluteError()]
    ):
    """ Wrap a Keras model as Tensorflow Federated model. 
    
    cf. https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#creating_a_model_with_keras
    """
    def _model():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        
        #keras_model = create_keras_model(
        #    nfeatures = nfeatures, compile = False#, **kwargs
        #    )
        
        keras_model = keras_creator()

        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec = (
                tf.TensorSpec((None, keras_model.input.shape[1]
                ), dtype = tf.float64),
                tf.TensorSpec((None,),           dtype = tf.float64)
            ), loss = loss, 
            metrics =  [
                tf.keras.metrics.MeanAbsoluteError()
                , tf.keras.metrics.MeanSquaredError()
                #, tfa.metrics.RSquare()
                #, tf.keras.metrics.R2Score() # only available in tf-nightly
                ]
        )

    return _model

def train_fed(
    model,
    train_data,
    eval_data = None,
    client_optimizer = lambda: tf.optimizers.Adam(learning_rate = .05),
    server_optimizer = lambda: tf.optimizers.Adam(learning_rate = .05),
    NUM_ROUNDS = 50,
    NUM_EPOCHS = 50,
    BATCH_SIZE = 128,
    SHUFFLE_BUFFER = 20,
    PREFETCH_BUFFER = 5,
    SEED = 42,
    verbose = True
    ):

    """
    FIXME: inputs and outputs
    
    cf. https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#evaluation
    """
    
    # if validation_split in arguments
    # eval setup
    #n_eval = [int(
    #    validation_split * 
    #    len(list(data.as_numpy_iterator()))
    #) for data in train_data]
    #train_data = [data.shuffle(10, seed = SEED, reshuffle_each_iteration=False) for data in train_data]
    #eval_data  = [train_data[i].take(n_eval[i]) for i in range(len(train_data))]
    #train_data = [train_data[i].skip(n_eval[i]) for i in range(len(train_data))]


    # prep the data
    train_data = [
        data.
            repeat(NUM_EPOCHS).
            shuffle(SHUFFLE_BUFFER, seed = SEED).
            batch(BATCH_SIZE).
            prefetch(PREFETCH_BUFFER)

      for data in train_data]
    
    # initialize the process (and evalation if any)

    process = tff.learning.algorithms.build_weighted_fed_avg(
        model,
        client_optimizer_fn = client_optimizer,
        server_optimizer_fn = server_optimizer)
    

    state = process.initialize()
    hist  = []

    if eval_data != None:
        eval_process  = tff.learning.algorithms.build_fed_eval(model)
        eval_state    = eval_process.initialize()
        model_weights = process.get_model_weights(state)
        eval_state    = eval_process.set_model_weights(eval_state, model_weights)
        _, perf_eval  = eval_process.next(eval_state, eval_data)

    # training
    for round in range(NUM_ROUNDS):
        
        # calculate the evaluation performance (before updating the model!)
        # Note:
        # - 'process' generally reflect the performance of the model at the beginning of the round, 
        # - if not calculated first, the evaluation metrics will always be one step ahead
        if eval_data != None:
            
            model_weights = process.get_model_weights(state)
            eval_state    = eval_process.set_model_weights(eval_state, model_weights)
            _, perf_eval  = eval_process.next(eval_state, eval_data)
            perf_eval = dict(perf_eval['client_work']['eval']['current_round_metrics'].items())
            
        
        # update FL process
        if SEED != None: tf.keras.utils.set_random_seed(SEED)
        state, perf = process.next(state, train_data)
        perf = dict(perf['client_work']['train'].items())

        # generate outputs
        if verbose: 
            print('====== ROUND {:2d} / {} ======'.format(round + 1, NUM_ROUNDS))
            print('TRAIN: {}'.format(perf))
            if eval_data != None: 
                print('EVAL:  {}'.format(perf_eval))

        
        # combine perf and perf_eval
        if eval_data != None:
            # rename keys in perf_eval
            perf_eval = {'val_' + key : val for key, val in  perf_eval.items()}
            # merge perf and perf_eval (> Python 3.9)
            perf = perf | perf_eval
        
        # extend history
        hist.append(perf)

    return {'process': process, 'history': hist, 'state': state}
