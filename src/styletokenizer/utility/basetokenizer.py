from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Sequence, Split, Metaspace
from tokenizers import Regex
from tokenizers import Tokenizer
from tokenizers.models import BPE

class Trainer:
    """
        see https://huggingface.co/docs/tokenizers/quicktour
    """

    def __init__(self, vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]):
        # Create a BPE tokenizer
        self.tokenizer = None
        # Initialize the trainer, specify the vocabulary size and special tokens
        self.trainer = None

    def set_pre_tokenizer(self, pre_tokenizer="ws"):
        if pre_tokenizer == "ws":
            # Use whitespace pre-tokenizer as default
            #   -> this ensures that the vocabulary in the end does not include common phrases like "it is"
            #   QUESTION: is this good for language variation?
            self.tokenizer.pre_tokenizer = Whitespace()
            print("Setting pre-tokenizer to Whitespace")
        elif pre_tokenizer == "byte":
            # GPT-specific regex
            #   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
            split = Split(pattern=Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
                          behavior="merged_with_next", invert=False)
            byte = ByteLevel(add_prefix_space=False, use_regex=False)
            self.tokenizer.pre_tokenizer = Sequence([split, byte])

            print("Setting pre-tokenizer to ByteLevel")
        elif pre_tokenizer == "meta":
            self.tokenizer.pre_tokenizer = Metaspace(replacement="▁",  prepend_scheme="never")
        elif pre_tokenizer is not None:
            raise ValueError(f"Invalid pre_tokenizer: {pre_tokenizer}")

    def train(self, sentences):
        # Train the tokenizer
        self.tokenizer.train_from_iterator(sentences, trainer=self.trainer)

    def tokenize(self, sentence):
        # Tokenize the provided sentence
        return self.tokenizer.encode(sentence).tokens