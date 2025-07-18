"""
    script to fit tokenizer
"""
import argparse
from tokenizers import Tokenizer, Regex, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split, Whitespace, WhitespaceSplit

import os

from styletokenizer.utility import datasets_helper
from styletokenizer.utility.tokenizer_vars import PRE_TOKENIZER, VOCAB_SIZE
from styletokenizer.utility.env_variables import at_uu, at_umich

# the tokenizer variables
if at_uu():
    FOLDER_BASE = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER"
elif at_umich():
    FOLDER_BASE = "/shared/3/projects/hiatus/TOKENIZER_wegmann"
else:
    FOLDER_BASE = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/"
OUT_PATH = os.path.join(FOLDER_BASE, "tokenizer")

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
from styletokenizer.fitting_corpora import CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED, CORPORA_PUBMED
from styletokenizer.train_data import UMICH_TRAIN_DATASET_PATH


def fit_tokenizer(fit_path: str, vocab_size: int, pre_tokenizer: str, dir_name: str, test=False):
    # init tokenizer with PRE_TOKENIZER
    tokenizer = init_tokenizer_with_regex(pre_tokenizer)
    # test if save works
    save_dir = f"{dir_name}/tokenizer.json"
    print(f"Will save tokenizer to: {save_dir}")

    # Initialize the BPE trainer with VOCAB_SIZE
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # Note: initial alphabet was not set for majority of experiments -> this can lead to not all 255 bytes being in the vocab ...
    #   https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py
    # BUT since they were (except for PubMed) all in the downstream tasks this should be fine

    # Get the text generator for the FITTING CORPUS
    text_generator = datasets_helper.train_text_generator(fit_path)

    # TRAIN the tokenizer on the corpus
    tokenizer.train_from_iterator(text_generator, trainer=trainer)

    os.makedirs(dir_name, exist_ok=True)
    # SAVE the tokenizer to the specified directory
    tokenizer.save(save_dir)
    return tokenizer


def init_tokenizer_with_regex(pre_tokenizer):
    # Initialize the BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # select pre-tokenizer
    # use byte level encoding
    #   --> with use_regex=False this is only transforming to byte level encoding
    #       importantly: it is performing no additional split as tested in Pre_Tokenizers.ipynb
    byte = ByteLevel(add_prefix_space=False, use_regex=False)
    # PRE-TOKENIZATION variable
    if (pre_tokenizer == "ws") or (pre_tokenizer == "gpt2") or (pre_tokenizer == "llama3"):
        if pre_tokenizer == "ws":
            # this might be different to what you expect, it is not equivalent to Whitespace(),
            #   which would be inverted r"\w+|[^\w\s]+"
            # instead this is already "a step towards" more gpt2/llama 3 like pre-tokenization,
            #   where the leading whitespace in front of non-whitespace characters is kept,
            #   otherwise all whitespaces are split off separately
            # in detail:
            #       (?!\S)  -->  negative lookahead for non-whitespace character
            #       \s+     -->  one or more whitespace characters
            regex_pattern = (
                r"\s+(?!\S)|\s+"
            )
            behavior = "merged_with_next"
        elif pre_tokenizer == "gpt2":
            # regex pattern copied from
            #   https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/data/encoders/gpt2_bpe_utils.py#L70
            regex_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            behavior = "isolated"
        else:
            # regex pattern copied from
            #   https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L47
            regex_pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+["
                             r"\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
            behavior = "isolated"

        split = Split(pattern=Regex(regex_pattern), behavior=behavior, invert=False)
        pre_tokenizer = Sequence([split, byte])
    elif pre_tokenizer == "no":
        pre_tokenizer = byte
    elif pre_tokenizer == "wsorg":
        regex_pattern = (
            r"\s+"
        )
        split = Split(pattern=Regex(regex_pattern), behavior="isolated", invert=False)
        pre_tokenizer = Sequence([split, byte])
    else:
        raise ValueError(f"Invalid pre-tokenizer: {pre_tokenizer}")
    tokenizer.pre_tokenizer = pre_tokenizer
    # https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py
    tokenizer.decoder = decoders.ByteLevel()  # originally did not include this, but: only about decoding anyway
    tokenizer.post_processor = processors.ByteLevel()  # originally did not include this, but: only about decoding anyway
    return tokenizer


def main(fitting_corpus_path: str, vocab_size: int, pre_tokenize: str, test=False):
    # get base dir of fit_path
    corpus_name = os.path.basename(fitting_corpus_path)
    # set the output directory name for tokenizer
    dir_name = f"{OUT_PATH}/{corpus_name}-{pre_tokenize}-{vocab_size}"
    # check if OUT PATH exists
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(f"OUT_PATH does not exist: {OUT_PATH}")

    fit_tokenizer(fitting_corpus_path, vocab_size, pre_tokenize, dir_name, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    # Add options to the mutually exclusive group
    group.add_argument("--twitter", action="store_true", help="Use Twitter as the fitting corpus.")
    group.add_argument("--wikipedia", action="store_true", help="Use Wikipedia as the fitting corpus.")
    group.add_argument("--mixed", action="store_true", help="Use a mixed corpus for fitting.")
    group.add_argument("--webbook", action="store_true", help="Use a mixed corpus for fitting.")
    group.add_argument('--pubmed', action='store_true', help='Use the PubMed corpus for fitting.')

    # Define the valid vocabulary sizes
    vocab_sizes = VOCAB_SIZE
    # Add the vocab_size argument with restricted choices
    parser.add_argument("--vocab_size", type=int, choices=vocab_sizes, required=True,
                        help="Specify the vocabulary size. Must be one of: " + ", ".join(map(str, vocab_sizes)))

    # Add the pre_tokenize argument with choices
    parser.add_argument("--pre_tokenize", type=str, choices=PRE_TOKENIZER, required=True,
                        help="Specify the pre-tokenization method. Must be one of: 'no', 'ws', 'gpt2', 'llama3'.")
    # parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # Output based on selected options
    if args.twitter:
        print("Fitting corpus: Twitter")
        fit_path = CORPORA_TWITTER
    elif args.wikipedia:
        print("Fitting corpus: Wikipedia")
        fit_path = CORPORA_WIKIPEDIA
    elif args.mixed:
        print("Fitting corpus: Mixed")
        fit_path = CORPORA_MIXED
    elif args.webbook:
        print("Fitting corpus: WebBook")
        fit_path = UMICH_TRAIN_DATASET_PATH
    elif args.pubmed:
        print("Fitting corpus: PubMed")
        fit_path = CORPORA_PUBMED
    # Output the selected vocabulary size
    print(f"Vocabulary size: {args.vocab_size}")
    # Output the pre-tokenization method
    print(f"Pre-tokenization method: {args.pre_tokenize}")

    main(fitting_corpus_path=fit_path, vocab_size=args.vocab_size, pre_tokenize=args.pre_tokenize)
