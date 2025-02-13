
from typing import List, Dict

import numpy as np
import tokenization_scorer
import sys

from styletokenizer.utility.tokenizer_vars import (get_pretokenizer_paths, get_sorted_vocabularies_per_tokenizer,
                                                   get_tokenizer_name_from_path, get_corpus_paths, get_vocab_paths,
                                                   get_tokenizer_from_path, get_all_paths)
from styletokenizer.fitting_corpora import CORPORA_MIXED, CORPORA_TWITTER, CORPORA_WIKIPEDIA
from styletokenizer.utility import datasets_helper
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
# from tokenization_scorer.metrics import _seq_len as seq_len

from utility.webbook import CORPORA_WEBBOOK
import tqdm


def calc_renyi_efficency_from_generator(text_generator, tokenizer_path, power=2.5):
    tok_gen = tok_generator(text_generator, tokenizer_path=tokenizer_path)
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
    return tokenization_scorer.score(tok_gen, metric="renyi",
                                     power=power)  # 2.5 ideal in Exp 1.1 in https://aclanthology.org/2023.acl-long.284v2.pdf


def calc_sim_renyi_efficiency_from_generator(text_generator, tokenizer_path, sim_vocab_size, power=2.5):
    tok_gen = tok_generator(text_generator, tokenizer_path=tokenizer_path)
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
    return _renyi_sim(tok_gen, sim_vocab_size=sim_vocab_size, power=power)


def calc_seq_len_from_path(tokenizer_path, data_path):
    text_generator = datasets_helper.efficient_split_generator(data_path, split="dev")
    return calc_seq_len_from_generator(text_generator, tokenizer_path)


def calc_seq_len_from_generator(text_generator, tokenizer_path):
    tok_gen = tok_generator(text_generator, tokenizer_path=tokenizer_path)
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
    return tokenization_scorer.score(tok_gen, metric="seq_len")


def tok_generator(text_generator, tokenizer_path):
    tokenizer = get_tokenizer_from_path(tokenizer_path)
    for text in text_generator:
        yield " ".join(tokenizer.encode(text).tokens)




def _renyi_sim(text, sim_vocab_size, power=2.5) -> float:
    vocab_size, word_probs = _get_vocabsize_and_dist(text)
    if vocab_size < sim_vocab_size:
        raise ValueError(f"Actual vocabulary size {vocab_size} is smaller than "
                         f"the simulated vocabulary size {sim_vocab_size}")

    word_probs = np.array(word_probs)
    sorted_probs = np.sort(word_probs)[::-1]



    trimmed_probs = sorted_probs[:sim_vocab_size]
    word_probs = trimmed_probs / np.sum(trimmed_probs)
    print(f"Actual vocabulary size {vocab_size}, with simulated vocabulary size {sim_vocab_size}",
          file=sys.stderr)

    scale = 1 / (1 - power)

    val = scale * np.log2(np.sum(
        np.array(word_probs) ** power
    )) / np.log2(sim_vocab_size)
    return val


def _get_vocabsize_and_dist(text):
    import itertools
    if type(text) == str:
        text = [l.split() for l in tqdm.tqdm(text.split("\n"))]
        line_count = len(text)
    else:
        text, peekable1, peekable2 = itertools.tee(text, 3)
        line_count = 1
        if type(next(peekable1)) != str:
            line_count = len(list(peekable2))
            # flatten once more
            text = (w for l in text for w in tqdm.tqdm(l))
        text = (l.rstrip("\n").split() for l in tqdm.tqdm(text))
    # cleanup (remove empty lines and words)
    text = (
        (w.strip() for w in l if w.strip())
        for l in text
    )
    text = (l for l in text if l)
    _, word_probs, vocab_size = tokenization_scorer.metrics.get_prob_distribution(text)
    return vocab_size, word_probs
