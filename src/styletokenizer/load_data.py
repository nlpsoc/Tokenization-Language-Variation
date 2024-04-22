
import pickle
import html


def load_pickle_file(file_path):
    """
        load data
        unescapes html entites (especially the quoting on reddit with > or &gt;)
        function generated with CoPilot, April 22nd 2024
    :param file_path:
    :return:
    """
    pickled_data = []
    with open(file_path, 'rb') as file:
        while True:
            try:
                item = pickle.load(file)
                for key in item:
                    if isinstance(item[key], str):  # Only unescape string types
                        item[key] = html.unescape(item[key])
                pickled_data.append(item)
            except EOFError:
                break
    return pickled_data