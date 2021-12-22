

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
#TRAIN = "data/har_train_25-50.csv" # use seq len 25 or 50
#TEST = "data/har_test_25-50.csv" # use seq len 25 or 50
#VALID =  "" # optional, if none given test set will be used for validation

#TRAIN = "data/har_train_35.csv" # use seq len 35
#TEST = "data/har_test_35.csv" # use seq len 35
#VALID =  "" # optional, if none given test set will be used for validation

TRAIN = "data/mobi_fall_train.csv" # 
TEST = "data/mobi_fall_test.csv" # 
VALID =  "" # optional, if none given test set will be used for validation

# ###############################
# Hyperparameters
# ###############################

# General
#BATCH_SIZE = 32
SEQLENGTH = 200
EPOCHS = 100

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
