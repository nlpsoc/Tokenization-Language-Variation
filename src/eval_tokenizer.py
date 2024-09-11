from typing import List, Dict

from styletokenizer.utility.tokenizer_vars import (get_pretokenizer_paths, get_sorted_vocabularies_per_tokenizer,
                                                   get_tokenizer_name_from_path, get_corpus_paths, get_vocab_paths,
                                                   get_tokenizer_from_path)
from styletokenizer.fitting_corpora import CORPORA_MIXED, CORPORA_TWITTER, CORPORA_WIKIPEDIA
from styletokenizer.utility import datasets_helper
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


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
    get_comparative_tok_stats(vocab_paths)


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


def calc_renyi_efficiency(tokenizer_path, data_path):
    import tokenization_scorer
    text_generator = tok_generator(data_path, split="dev", tokenizer_path=tokenizer_path)
    return tokenization_scorer.score(text_generator, metric="renyi", power=2.5)


def tok_generator(dataset_path, split, tokenizer_path):
    tokenizer = get_tokenizer_from_path(tokenizer_path)
    text_generator = datasets_helper.train_text_generator(dataset_path, split=split)
    for text in text_generator:  # TODO: how to do tokenize in the right way again?, check that this works see test (!)
        yield tokenizer.encode(text).tokens
