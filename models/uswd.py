from config import WINDOW_SIZE, NUM_CLASSES
import torch
import torch.nn as nn


class USWDNet(nn.Module) :
    def __init__(self) :
        super(USWDNet, self).__init__()