import os
from typing import List, Dict

import tokenization_scorer

from styletokenizer.utility.tokenizer_vars import (get_pretokenizer_paths, get_sorted_vocabularies_per_tokenizer,
                                                   get_tokenizer_name_from_path, get_corpus_paths, get_vocab_paths,
                                                   get_tokenizer_from_path, get_all_paths)
from styletokenizer.fitting_corpora import CORPORA_MIXED, CORPORA_TWITTER, CORPORA_WIKIPEDIA
from styletokenizer.utility import datasets_helper
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from tokenization_scorer.metrics import _seq_len as seq_len

from styletokenizer.utility.preptraining_corpora import CORPORA_WEBBOOK
import tqdm


def get_most_middle_least_common_tokens(tokens):
    n = len(tokens)
    return {
        "most_common": tokens[:10],
        "middle": tokens[n // 2 - 5: n // 2 + 4],
        "least_common": tokens[-10:]
    }


def get_unique_tokens_per_tokenizer(vocabularies: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
        for a given dict of vocabularies, calculate the truly unique tokens for each tokenizer
    :param vocabularies:
    :return:
    """
    # Calculate truly unique tokens for each tokenizer
    unique_tokens = {}
    for name, vocab_list in vocabularies.items():
        # Union of vocabularies from the other tokenizers
        others = set.union(*(set(vocabularies[other_name]) for other_name in vocabularies if other_name != name))

        # Subtract the other vocabularies from the current one to get truly unique tokens
        unique_tokens[name] = [token for token in vocab_list if token not in others]

    return unique_tokens


def main():
    pretokenizer_paths = get_pretokenizer_paths()
    get_comparative_tok_stats(pretokenizer_paths)
    corpus_paths = get_corpus_paths()
    get_comparative_tok_stats(corpus_paths)
    vocab_paths = get_vocab_paths()
    get_consecutive_tok_stats(vocab_paths)

    # get Renyi and Toks/word for each tokenizer
    for tokenizer_path in get_all_paths():
        print(f"\n\n{tokenizer_path}")
        for corpus_path in [CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED, CORPORA_WEBBOOK]:
            print(f"\n{corpus_path}")
            print(f"Percentile Frequency: {calc_precentile_freq(tokenizer_path, corpus_path)}")
            print(f"Sequence Length: {calc_seq_len(tokenizer_path, corpus_path)}")
            print(f"Renyi Efficiency: {calc_renyi_efficiency_from_path(tokenizer_path, corpus_path)}")
            print(f"Sequence Length: {calc_seq_len(tokenizer_path, corpus_path)}")
            print(f"Average Tokens per Word: {calc_avg_tok_per_word_from_path(tokenizer_path, corpus_path)}")


def get_comparative_tok_stats(tokenizer_paths):
    tokenizer_names = [get_tokenizer_name_from_path(path) for path in tokenizer_paths]
    vocabularies = get_sorted_vocabularies_per_tokenizer(tokenizer_paths)
    unique_tokens = get_unique_tokens_per_tokenizer(vocabularies)
    results = {
        'vocab_sizes': {name: len(vocab_list) for name, vocab_list in vocabularies.items()},
        'common_token_count_all': len(set.intersection(*(set(vocabularies[name]) for name in vocabularies))),
        'unique_tokens_count': {name: len(tokens) for name, tokens in unique_tokens.items()},
        'unique_tokens_examples': {name: get_most_middle_least_common_tokens(tokens)
                                   for name, tokens in unique_tokens.items()},
        'pairwise_common_tokens': {f"{tokenizer_names[i]}": {f"{tokenizer_names[j]}":
            len(set(
                vocabularies[tokenizer_names[i]]).intersection(
                set(vocabularies[tokenizer_names[j]]))) for j in range(i + 1, len(tokenizer_names))}
            for i in range(len(tokenizer_names))},
        'unique_tokens': unique_tokens
    }

    # Print the results
    print("Vocabulary Sizes:")
    for name, size in results['vocab_sizes'].items():
        print(f"{name}: {size} tokens")

    print(f"\nCommon Tokens Across All: {results['common_token_count_all']}")

    print("\nTruly Unique Tokens with Frequency Examples:")
    for name, tokens in results['unique_tokens'].items():
        print(f"{name}: {len(tokens)} unique tokens")
        examples = results['unique_tokens_examples'][name]
        print(f"Most Common: {examples['most_common']}")
        print(f"Middle: {examples['middle']}")
        print(f"Least Common: {examples['least_common']}")

    print("\nPairwise Common Tokens:")
    for i in range(len(tokenizer_names)):
        for j in range(i + 1, len(tokenizer_names)):
            print(f"{tokenizer_names[i]} & {tokenizer_names[j]}: "
                  f"{results['pairwise_common_tokens'][tokenizer_names[i]][tokenizer_names[j]]} common tokens")

    # Plot the Venn diagram only if there are 3 tokenizers
    if len(tokenizer_names) == 3:
        plt.figure()
        venn_params = []
        for i in range(len(tokenizer_names)):
            if i == 0:
                venn_params.append(results['unique_tokens_count'][tokenizer_names[i]])
            for j in range(i + 1, len(tokenizer_names)):
                if i == 0:
                    venn_params.append(results['unique_tokens_count'][tokenizer_names[j]])
                venn_params.append(results['pairwise_common_tokens'][tokenizer_names[i]][tokenizer_names[j]])
        venn_params.append(results['common_token_count_all'])

        # per two tokenizers:
        venn3(subsets=venn_params, set_labels=tokenizer_names)
        plt.show()

    return results


def get_consecutive_tok_stats(tokenizer_paths):
    vocabularies = get_sorted_vocabularies_per_tokenizer(tokenizer_paths)

    # Calculate the overlap for consecutive tokenizers and find newly added tokens
    overlap_ratios = {}
    added_tokens_examples = {}
    folder_names = list(vocabularies.keys())

    for i in range(len(folder_names) - 1):
        name_1 = folder_names[i]
        name_2 = folder_names[i + 1]
        print(f"At pair {(name_1, name_2)}")

        # Get vocabularies as ordered lists
        vocab_1 = vocabularies[name_1]
        vocab_2 = vocabularies[name_2]

        # Calculate the overlap
        overlap = len(set(vocab_1).intersection(set(vocab_2)))
        added = set(vocab_2) - set(vocab_1)
        smaller_vocab_size = min(len(vocab_1), len(vocab_2))

        # Calculate the overlap ratio
        overlap_ratio = overlap / smaller_vocab_size
        overlap_ratios[f"{name_1} & {name_2}"] = overlap_ratio

        print("Getting examples of newly added tokens.")
        # sort new tokens by frequency
        added_tokens = [token for token in vocab_2 if token in added]
        n = len(added_tokens)
        print(added_tokens[:100])

        # Get examples of the most frequent, middle, and least frequent added tokens
        examples = {
            "most_frequent": added_tokens[:10],
            "middle": added_tokens[max(0, n // 2 - 5):max(0, n // 2 + 5)],
            "least_frequent": added_tokens[-10:]
        }
        added_tokens_examples[f"{name_1} -> {name_2}"] = examples

    # Print the results
    print("Overlap Ratios between Consecutive Tokenizers:")
    for pair, ratio in overlap_ratios.items():
        print(f"{pair}: {ratio:.2f}")

    print("\nExamples of Added Tokens:")
    vocab_1 = vocabularies[folder_names[0]]
    print(f"Tokens in {folder_names[0]}")
    print(f"Most Frequent: {vocab_1[:15]}")
    print(f"Middle: {vocab_1[len(vocab_1) // 2 - 5: len(vocab_1) // 2 + 5]}")
    print(f"Least Frequent: {vocab_1[-10:]}")
    for transition, examples in added_tokens_examples.items():
        print(f"\nTransition: {transition}")
        print("Most Frequent Added Tokens:", examples['most_frequent'])
        print("Middle Added Tokens:", examples['middle'])
        print("Least Frequent Added Tokens:", examples['least_frequent'])


def calc_renyi_efficiency_from_path(tokenizer_path, data_path):
    text_generator = tok_generator(data_path, split="dev", tokenizer_path=tokenizer_path)
    return calc_renyi_efficency_from_generator(text_generator)


def calc_renyi_efficency_from_generator(text_generator):
    import tokenization_scorer
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
    return tokenization_scorer.score(text_generator, metric="renyi", power=2.5)


def calc_precentile_freq(tokenizer_path, data_path):
    text_generator = tok_generator(data_path, split="dev", tokenizer_path=tokenizer_path)
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
    return tokenization_scorer.score(text_generator, metric="percentile_freq", perc_start=0.03, perc_end=0.83)


def calc_seq_len(tokenizer_path, data_path):
    import tokenization_scorer
    text_generator = tok_generator(data_path, split="dev", tokenizer_path=tokenizer_path)
    return seq_len(list(text_generator))


def calc_avg_tok_per_word_from_path(tokenizer_path, data_path):
    text_generator = datasets_helper.efficient_split_generator(data_path, split="dev")
    return calc_avg_tok_from_generator(text_generator, tokenizer_path)


def calc_avg_tok_from_generator(text_generator, tokenizer_path):
    tokenizer = get_tokenizer_from_path(tokenizer_path)
    nbr_tokens = 0
    nbr_words = 0
    for text in text_generator:
        tokens = tokenizer.encode(text).tokens
        nbr_words += len(text.split())
        nbr_tokens += len(tokens)
    return nbr_tokens / nbr_words


def tok_generator(dataset_path, split, tokenizer_path):
    tokenizer = get_tokenizer_from_path(tokenizer_path)
    text_generator = datasets_helper.efficient_split_generator(dataset_path, split=split)
    for text in text_generator:  # TODO: how to do tokenize in the right way again?, check that this works see test (!)
        yield tokenizer.encode(text).tokens
