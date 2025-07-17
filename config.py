# params
SEED = 42
EPOCHS = 50
NUM_WORKERS = 4
BATCH_SIZE = 32
LOSS = "focal" # focal
LR = 0.004

# models
OUT_DIM = 256
EMB_DIM = 128
DROPOUT_SCORE = 0.2
ENCODER_TYPE = "c" # c (CNN) / t (Transformer)
MODEL_NAME = "uswd_s" # uswd_s (seperate encoder) / uswd_u (union encoder) / lstm / cnn / mlp

# transformers
N_HEAD = 4
NUM_LAYERS = 3
DROPOUT_TR = 0.1

# data
WINDOW_SIZE = 100
NUM_CLASSES = 10
DATA_ROOT = "./data/dataset/ultrasonic"
LOG_PATH = "./logs"
DEVICES = [0, 1]