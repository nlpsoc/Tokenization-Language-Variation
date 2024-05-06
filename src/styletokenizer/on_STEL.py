import sys

import numpy as np

from src.styletokenizer import tokenizer
from sklearn.metrics.pairwise import cosine_similarity

# add STEL folder to path
sys.path.append('../STEL/src/')

from STEL import STEL
from STEL.similarity import Similarity


class TokenizerSimilarity(Similarity):
    def __init__(self, tok=None):
        super().__init__()
        if tok is None:
            self.tokenizer = tokenizer.get("whitespace")
        else:
            self.tokenizer = tok

    def similarity(self, s1, s2):
        s1 = self.tokenizer(s1)
        s2 = self.tokenizer(s2)

        vocabulary = list(set(s1 + s2))
        s1 = one_hot_encode(s1, vocabulary)
        s2 = one_hot_encode(s2, vocabulary)
        return cosine_similarity([s1], [s2])[0][0]


def one_hot_encode(words, vocabulary):
    encoding = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            encoding[vocabulary.index(word)] = 1
    return encoding


def eval_tokenizer(tok):
    STEL.eval_on_STEL(style_objects=[TokenizerSimilarity(tok)])

roberta = "roberta-base"
t5="google-t5/t5-base"
xlmroberta = "xlm-roberta-base"
llama27b = "meta-llama/Llama-2-7b-hf"
mixtreal = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama38b = "meta-llama/Meta-Llama-3-8B"
anna = "AnnaWegmann/Style-Embedding"

eval_tokenizer(tok=tokenizer.get(roberta))  # "FacebookAI/xlm-roberta-base"
