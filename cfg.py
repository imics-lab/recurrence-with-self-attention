"""
Run Time Configuration

Authors: Alexander Katrompas, Theodoros Ntakouris, Vangelis Metsis
Organization: Texas State University

"""

# ###############################
# command line parameter defaults
# ###############################
MODEL = 1 # set to 1 or 2
MODELS = [1,2]
VERBOSE = False
GRAPH = False

# ###############################
# data files
# ###############################
TRAIN = "data/mobi_fall_train.csv" # 
TEST = "data/mobi_fall_test.csv" # 
VALID =  "" # optional, if none given test set will be used for validation

# ###############################
# Hyperparameters
# ###############################

# General
#BATCH_SIZE = 32
SEQLENGTH = 200
EPOCHS = 2

# LSTM
LSTM = 256
DENSE1 = 128
DENSE2 = 64
OUTPUT = 1
DROPOUT = 0.25
LSTM_ATTENTION = True

# Transformer
NUM_HEADS = 4
HEAD_SIZE = 32
FF_DIM = 32
NUM_ATTN_LAYERS = 4
MLP_DIMS = [16, 8] # can be []
