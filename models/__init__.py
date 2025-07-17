from config import MODEL_NAME
from baselines import MLP, CNN, LSTM
from uswd import USWDNet_Uni, USWDNet_Sep

def get_model() :
    if MODEL_NAME == "uswd_s" :
        return USWDNet_Sep()
    elif MODEL_NAME == "uswd_u" :
        return USWDNet_Uni()
    elif MODEL_NAME == "lstm" :
        return LSTM()
    elif MODEL_NAME == "cnn" :
        return CNN()
    else :
        return MLP()