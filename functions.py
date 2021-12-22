"""
Supporting functions

Author: Alex katrompas
"""

# python libs
import sys
import re
import pandas as pd
import numpy as np
from tensorflow.keras import layers

# local libs
import cfg
from transformer_encoder import TransformerEncoder

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
         num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res    

def print_setup(model, verbose, graph):
    print()
    print("Executing...")
    if model == 1:
        print(" - LSTM")
    elif model == 2:
        print(" - Transformer")
    if verbose:
        print(" - Verbose: On")
    else:
        print(" - Verbose: Off")
    if graph:
        print(" - Graphing: On")
    else:
        print(" - Graphing: Off")
    print()

def get_args():
    # set optional defaults in case of error or no parameters
    model = cfg.MODEL
    verbose = cfg.VERBOSE
    graph = cfg.GRAPH
    model_set = False
    switches_set = False
    
    argc = len(sys.argv) # get command line arguments
    
    if argc == 2 or argc == 3: # correct number of parameters
        
        if re.search("^[1-2]$", sys.argv[1]):
            model = int(sys.argv[1])
            model_set = True
        elif re.search("^-[vg]+", sys.argv[1]):
            if 'v' in sys.argv[1]:
                verbose = True
            if 'g' in sys.argv[1]:
                graph = True
            switches_set = True
        if argc == 3:
            if re.search("^[1-2]$", sys.argv[2]) and not model_set:
                model = int(sys.argv[2])
            elif re.search("^-[vg]+", sys.argv[2]) and not switches_set:
                if 'v' in sys.argv[2]:
                    verbose = True
                if 'g' in sys.argv[2]:
                    graph = True
    print_setup(model, verbose, graph)
    return model, verbose, graph

def normalize(data, np_array=False, scaled=False):
    """
    Normalize and optionally scale a dataset.
    
    Parameters: Pandas DataFrame or Numpy,
                optional np_array flag for return type, optional scale flag
    Processing: Will normalize and optionally scale the dataset
    Return: Pandas DataFrame or Numpy array with normalized/scaled data
    """
    # ensure floats
    data = data.astype(float)
    
    if detect_datatype(data) == DataType.NUMPY:
        # set-up normalization
        high = 1.0
        low = 0.0
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        rng = maxs - mins
        # normalize
        data = high - (((high - low) * (maxs - data)) / rng)
        # scale if needed
        if scaled:
            data = (data / .5) - 1
    elif detect_datatype(data) == DataType.DATAFRAME:
        # normalize
        [data[col].update((data[col] - data[col].min()) / (data[col].max() - data[col].min())) for col in data.columns]
        # scale if needed
        if scaled:
            [data[col].update((data[col] / .5)-1) for col in data.columns]
    
    # return appropriate object
    if np_array and detect_datatype(data) == DataType.DATAFRAME:
        data = data.to_numpy()
    elif not np_array and detect_datatype(data) == DataType.NUMPY:
        data = pd.DataFrame(data)
    
    return data

def getio(data, x_cols, np_array=False):
    """
    Slice a dataset into input and output features based on the last x_cols
    
    Parameters: Pandas DataFrame or Numpy array, number of column from left to right
           that are the input columns.
    Processing: Will slice into two sets, input and output data
    Return: Two data sets of the same type sent in, input and output
    """
    total_cols = column_count(data)
    if len(data.shape) != 2:
        raise TypeError("Input data must be 2D.")
    elif x_cols < 1 or x_cols >= total_cols:
        raise ValueError("Input column count must be between 1 and " + str(total_cols - 1) + " inclusive")

    if detect_datatype(data) == DataType.NUMPY:
        X = data[:,:x_cols]
        Y = data[:, x_cols:]
    elif detect_datatype(data) == DataType.DATAFRAME:
        # left of the , ommitting start and stop gives "all rows"
        # right of the , ommitting start and including number of columns
        X = data.iloc[:,:x_cols]
        Y = data.iloc[:, x_cols:]
        
    if np_array and \
        (detect_datatype(X) == DataType.DATAFRAME and detect_datatype(Y) == DataType.DATAFRAME):
        X = X.to_numpy()
        Y = Y.to_numpy()
        

    return X, Y

def load_data(train_name, test_name, valid_name="", labels = 1, norm = False, index = None, header = None):
    train = pd.read_csv(train_name, index_col=index, header=header).astype(float, errors='ignore')
    test = pd.read_csv(test_name, index_col=index, header=header).astype(float, errors='ignore')
    if valid_name:
        valid = pd.read_csv(valid_name, index_col=index, header=header).astype(float, errors='ignore')
    
    # if number of features not defined, assume columns -1
    features = train.shape[1] - labels

    train_X = train.iloc[:, 0:features]
    train_Y = train.iloc[:,features:]
    del train

    test_X = test.iloc[:, 0:features]
    test_Y = test.iloc[:,features:]
    del test

    if valid_name:
        valid_X = valid.iloc[:, 0:features]
        valid_Y = valid.iloc[:,features:]
        del valid
    else:
        valid_X = test_X.copy()
        valid_Y = test_Y.copy()

    if norm:
        [train_X[col].update((train_X[col] - train_X[col].min()) / (train_X[col].max() - train_X[col].min())) for col in train_X.columns]
        [test_X[col].update((test_X[col] - test_X[col].min()) / (test_X[col].max() - test_X[col].min())) for col in test_X.columns]
        [valid_X[col].update((valid_X[col] - valid_X[col].min()) / (valid_X[col].max() - valid_X[col].min())) for col in valid_X.columns]

    return train_X.to_numpy(), train_Y.to_numpy(), test_X.to_numpy(), test_Y.to_numpy(), valid_X.to_numpy(),  valid_Y.to_numpy()

def shape_3D_data(data, timesteps):
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
    """
    Resape 2D data into 3D data of groups of 2D timesteps
    
    Parameters: Pandas DataFrame or Numpy 3D array, number of timesteps/group
    Processing: Reshape 2D data into 3D data of array of 2D 
    Return: The reshaped data as numpy 3D array
    """

    # samples are total number of input vectors
    samples = data.shape[0]
    # time steps are steps per batch
    features = data.shape[1]
    
    # samples must divide evenly by timesteps to create an even set of batches
    if not(samples % timesteps):
        return np.array(data).reshape(int(data.shape[0] / timesteps), timesteps, features)
    else:
        msg = "timesteps must divide evenly into total samples: " + str(samples) + "/" \
            + str(timesteps) + "=" + str(round(float(samples) / float(timesteps), 2))
        raise ValueError(msg)
   