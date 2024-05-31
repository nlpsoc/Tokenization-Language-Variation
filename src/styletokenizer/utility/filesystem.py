import logging
import os
import sys
import random
import numpy


def get_dir_to_src():
    """
        get the path to the src root directory
    :return:
    """
    dir_path = os.path.dirname(os.path.normpath(__file__))
    base_dir = os.path.basename(dir_path)
    if base_dir == "utility":
        return os.path.dirname(os.path.dirname(dir_path))
    elif base_dir == "styletokenizer":
        return os.path.dirname(dir_path)
    else:
        return dir_path


def get_data_dir(jian=True):
    # test if on local machine
    if not on_cluster():
        return get_dir_to_src() + "/.."
    elif jian:
        return "/shared/3/projects/hiatus/TOKENIZER_wegmann/jian"
    else:
        return "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/"


def on_cluster():
    return "git projects.nosync" not in get_dir_to_src()



def set_global_seed(seed=42, w_torch=True):
    """
    Make calculations reproducible by setting RANDOM seeds
    :param seed:
    :param w_torch:
    :return:
    """
    # set the global variable to the new var throughout
    global SEED
    SEED = seed
    if 'torch' not in sys.modules:
        w_torch = False
    if w_torch:
        import torch
        logging.info(f"Running in deterministic mode with seed {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
