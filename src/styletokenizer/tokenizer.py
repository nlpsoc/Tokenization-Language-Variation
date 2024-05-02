from utility.torchtokenizer import TorchTokenizer
from utility import whitespace, POS


def get(tokenizer_name):
    print(f"Setting tokenizer to {tokenizer_name}")
    if tokenizer_name == "whitespace":
        return whitespace.tokenize
    elif tokenizer_name == "POS":
        return POS.tokenize
    elif tokenizer_name:
        return TorchTokenizer(tokenizer_name).tokenize

# def get_vocab(tokenizer):
#     if tokenizer_name == "whitespace":
#         return whitespace.vocab
#     elif tokenizer_name == "POS":
#         return POS.vocab
#     elif tokenizer_name:
#         return TorchTokenizer(tokenizer_name).vocab