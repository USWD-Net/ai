import torch
import numpy as np
import random
from config import SEED

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # 멀티 GPU를 사용하는 경우
    np.random.seed(SEED)
    random.seed(SEED)
    
    # CuDNN 결정론적 동작 설정 (성능 저하 가능성 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False