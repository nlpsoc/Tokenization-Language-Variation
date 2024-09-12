import os
from typing import Dict, List

from tokenizers import Tokenizer

from styletokenizer.utility.env_variables import at_uu, at_umich

PRE_TOKENIZER = ["ws", "gpt2", "llama3"]
VOCAB_SIZE = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
FITTING_CORPORA = ["twitter", "wikipedia", "mixed"]

DEFAULT_FITTING_CORPORA = "mixed"
DEFAULT_VOCAB_SIZE = 32000
DEFAULT_PRE_TOKENIZER = "gpt2"
if at_uu():
    OUT_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer"
elif at_umich():
    OUT_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/tokenizer"
else:
    OUT_PATH = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/tokenizer"


def get_name(corpus_name, pre_tokenizer, vocab_size):
    dir_name = f"{OUT_PATH}/{corpus_name}-{pre_tokenizer}-{vocab_size}/tokenizer.json"
    return dir_name


def get_pretokenizer_paths():
    return [get_name(DEFAULT_FITTING_CORPORA, pre_tokenizer, DEFAULT_VOCAB_SIZE) for pre_tokenizer in PRE_TOKENIZER]


def get_corpus_paths():
    return [get_name(corpus, DEFAULT_PRE_TOKENIZER, DEFAULT_VOCAB_SIZE) for corpus in FITTING_CORPORA]


def get_vocab_paths():
    return [get_name(DEFAULT_FITTING_CORPORA, DEFAULT_PRE_TOKENIZER, vocab_size) for vocab_size in VOCAB_SIZE]


def get_all_paths():
    return list(set(get_pretokenizer_paths() + get_corpus_paths() + get_vocab_paths()))


def get_tokenizer_name_from_path(path):
    return os.path.basename(os.path.dirname(path))


def get_sorted_vocabularies_per_tokenizer(tokenizer_paths) -> Dict[str, List[str]]:
    """
        Load tokenizers and extract vocabularies
    :param tokenizer_paths:
    :return: vocabularies,
    sorted by frequency (assuming tokenizers library stores IDs in order of frequency) in descending order
    """
    # Load tokenizers and extract vocabularies
    vocabularies = {}
    folder_names = []
    for path in tokenizer_paths:
        # Extract the last folder name from the path
        folder_name = get_tokenizer_name_from_path(path)
        folder_names.append(folder_name)
        tokenizer = Tokenizer.from_file(path)
        vocab = tokenizer.get_vocab()

        # Sort tokens based on their frequency (ID)
        sorted_tokens = sorted(vocab.items(), key=lambda item: item[1])

        # Store sorted tokens (by frequency) as a list of token strings
        vocabularies[folder_name] = [token for token, _ in sorted_tokens]
    return vocabularies


def get_tokenizer_from_path(path):
    return Tokenizer.from_file(path)
