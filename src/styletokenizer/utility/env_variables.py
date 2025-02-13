"""
    This file contains utility functions to set global variables and paths
"""
import logging
import os
import random
import sys

import numpy

from styletokenizer.utility.custom_logger import log_and_flush


def set_torch_device():
    import torch

    global device
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    elif torch.backends.mps.is_available():
        print('No GPU available, using the MPS backend for MacOS instead.')
        device = torch.device("mps")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


UMICH_CACHE_DIR = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
UU_CACHE_DIR = "/hpc/uu_cs_nlpsoc/02-awegmann/huggingface"


def set_cache():
    if at_uu():
        log_and_flush("Using UU cluster cache")
        os.environ["HF_HOME"] = UU_CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = os.path.join(UU_CACHE_DIR, "datasets")
        import datasets
        datasets.config.HF_DATASETS_CACHE = os.path.join(UU_CACHE_DIR, "datasets")
        os.environ["WANDB_CACHE_DIR"] = '/hpc/uu_cs_nlpsoc/02-awegmann/wandb_cache'
        return UU_CACHE_DIR
    elif at_umich():
        log_and_flush("Using UMich cluster cache")
        os.environ["HF_HOME"] = UMICH_CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = UMICH_CACHE_DIR
        return UMICH_CACHE_DIR
    else:
        log_and_flush("Using local cache")
        return None


def at_uu() -> bool:
    return "uu_cs_nlpsoc" in os.getcwd()


def at_umich() -> bool:
    return "annaweg" in os.getcwd()

def at_local() -> bool:
    return "anna/Documents/git projects.nosync" in os.getcwd()


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
