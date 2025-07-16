# params
SEED = 42
NUM_WORKERS = 4
BATCH_SIZE = 32
LOSS = "focal" # focal , ce

# models
OUT_DIM = 256
EMB_DIM = 128
DROPOUT_SCORE = 0.2
ENCODER_TYPE = "c" # c (CNN) / t (Transformer)

# transformers
N_HEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.1

# data
WINDOW_SIZE = 100
NUM_CLASSES = 10