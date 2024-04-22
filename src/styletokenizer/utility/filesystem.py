import os


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