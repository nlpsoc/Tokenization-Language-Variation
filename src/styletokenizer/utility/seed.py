import random
import numpy as np
import torch
from sklearn.utils import check_random_state


def set_global():
    print("Setting global seed to 123")
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    check_random_state(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
