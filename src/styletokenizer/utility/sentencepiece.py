"""
    sentence piece is BPE with including " " as a character, i.e., just a changed pre-tokenizer

    https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py

"""
from styletokenizer.utility import basetokenizer
from tokenizers import Tokenizer
from tokenizers.implementations.sentencepiece_bpe import SentencePieceBPETokenizer

