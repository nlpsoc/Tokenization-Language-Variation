from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class TrainTokenizer:

    def __init__(self):
        # Create a BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    def train(self, sentences, vocab_size=30000):
        # Initialize the trainer, specify the vocabulary size and special tokens
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        # Use whitespace pre-tokenizer
        self.tokenizer.pre_tokenizer = Whitespace()

        # Train the tokenizer
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)

    def tokenize(self, sentence):
        # Tokenize the provided sentence
        return self.tokenizer.encode(sentence).tokens
