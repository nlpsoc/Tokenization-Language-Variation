import torch
import logging


def set_torch_device():
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


def set_logging():
    """
    set logging format for calling logging.info
    :return:
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    hdlr = root.handlers[0]
    fmt = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    hdlr.setFormatter(fmt)

