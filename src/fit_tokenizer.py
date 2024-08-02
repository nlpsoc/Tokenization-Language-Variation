import argparse
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence, Split, PreTokenizer

import os

from styletokenizer.utility import datasets_helper

OUT_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/tokenizer"

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
from styletokenizer.fitting_corpora import CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED


def fit_tokenizer(fit_path: str, vocab_size: int, pre_tokenizer: str, dir_name: str, test=False):
    # init tokenizer with PRE_TOKENIZER
    tokenizer = init_tokenizer_with_regex(pre_tokenizer)
    # test if save works
    save_dir = f"{dir_name}/tokenizer.json"
    print(f"Saving tokenizer to: {save_dir}")
    tokenizer.save(save_dir)

    # Initialize the BPE trainer with VOCAB_SIZE
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    # Get the text generator for the FITTING CORPUS
    text_generator = datasets_helper.train_text_generator(fit_path)

    # TRAIN the tokenizer on the corpus
    tokenizer.train_from_iterator(text_generator, trainer=trainer)

    # SAVE the tokenizer to the specified directory
    os.makedirs(dir_name, exist_ok=True)
    tokenizer.save(save_dir)
    return tokenizer


def init_tokenizer_with_regex(pre_tokenizer):
    # Initialize the BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # PRE-TOKENIZATION variable
    if pre_tokenizer == "ws":
        split = Whitespace()
    elif (pre_tokenizer == "gpt2") or (pre_tokenizer == "llama3"):
        if pre_tokenizer == "gpt2":
            # regex pattern copied from
            #   https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/data/encoders/gpt2_bpe_utils.py#L70
            regex_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        else:
            # regex pattern copied from
            #   https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L47
            regex_pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+["
                             r"\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        split = Split(pattern=Regex(regex_pattern),
                      behavior="merged_with_next", invert=False)
    else:
        raise ValueError(f"Invalid pre-tokenizer: {pre_tokenizer}")
    # assert split is a variable
    assert split is not None
    # use byte level encoding
    #   --> with use_regex=False this is only transforming to byte level encoding
    #       importantly: it is performing no additional split as tested in Pre_Tokenizers.ipynb
    byte = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.pre_tokenizer = Sequence([split, byte])
    return tokenizer


def main(fitting_corpus_path: str, vocab_size: int, pre_tokenize: str, test=False):
    # get base dir of fit_path
    corpus_name = os.path.dirname(fitting_corpus_path)
    # set the output directory name for tokenizer
    dir_name = f"{OUT_PATH}/{corpus_name}-{pre_tokenize}-{vocab_size}"

    fit_tokenizer(fitting_corpus_path, vocab_size, pre_tokenize, dir_name, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    # Add options to the mutually exclusive group
    group.add_argument("--twitter", action="store_true", help="Use Twitter as the fitting corpus.")
    group.add_argument("--wikipedia", action="store_true", help="Use Wikipedia as the fitting corpus.")
    group.add_argument("--mixed", action="store_true", help="Use a mixed corpus for fitting.")

    # Define the valid vocabulary sizes
    vocab_sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
    # Add the vocab_size argument with restricted choices
    parser.add_argument("--vocab_size", type=int, choices=vocab_sizes, required=True,
                        help="Specify the vocabulary size. Must be one of: " + ", ".join(map(str, vocab_sizes)))

    # Add the pre_tokenize argument with choices
    parser.add_argument("--pre_tokenize", type=str, choices=["ws", "gpt2", "llama3"], required=True,
                        help="Specify the pre-tokenization method. Must be one of: 'ws', 'gpt2', 'llama3'.")
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
    # Output the selected vocabulary size
    print(f"Vocabulary size: {args.vocab_size}")
    # Output the pre-tokenization method
    print(f"Pre-tokenization method: {args.pre_tokenize}")

    main(fitting_corpus_path=fit_path, vocab_size=args.vocab_size, pre_tokenize=args.pre_tokenize)

# def fit_wiki_tokenizer(corpus_iterator, vocab_size, dir_name, test=False):
#     old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#     tokenizer = old_tokenizer.train_new_from_iterator(corpus_iterator, vocab_size=vocab_size,
#                                                       length=(6459000 if not test else 1000))
#     os.makedirs(dir_name, exist_ok=True)
#     tokenizer.save_pretrained(f"{dir_name}")
#     return tokenizer
