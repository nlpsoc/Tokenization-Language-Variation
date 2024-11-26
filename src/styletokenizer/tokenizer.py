import os

from styletokenizer.utility import whitespace, POS
from utility.env_variables import at_uu, at_umich


def get(tokenizer_name):
    from styletokenizer.utility.torchtokenizer import TorchTokenizer
    print(f"Setting tokenizer to {tokenizer_name}")
    if tokenizer_name == "whitespace":
        return whitespace.tokenize
    elif tokenizer_name == "POS":
        return POS.tokenize
    elif tokenizer_name:
        return TorchTokenizer(tokenizer_name).tokenize


if at_uu():
    BASE_TOKENIZER_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/"
elif at_umich():
    raise NotImplementedError("UMich not implemented")
    # BASE_TOKENIZER_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/"
else:
    BASE_TOKENIZER_PATH = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/tokenizer/"
TOKENIZER_PATHS = [
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "twitter-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "wikipedia-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-ws-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-llama3-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-500/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-1000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-2000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-4000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-8000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-16000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-64000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-128000/tokenizer.json"),
]
