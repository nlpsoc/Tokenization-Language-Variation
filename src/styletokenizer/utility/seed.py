import random
import numpy as np
import torch
from sklearn.utils import check_random_state

from styletokenizer.utility.custom_logger import log_and_flush


def set_global(seed=123):
    log_and_flush(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    check_random_state(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
