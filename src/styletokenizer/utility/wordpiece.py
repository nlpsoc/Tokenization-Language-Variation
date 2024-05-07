from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import styletokenizer.utility.basetokenizer as basetokenizer


class WordpieceTokenizer(basetokenizer.Trainer):
    """
        see https://huggingface.co/docs/tokenizers/quicktour
    """

    def __init__(self, vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                 pre_tokenizer="ws"):
        super().__init__(vocab_size, special_tokens)
        # Create a BPE tokenizer
        self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        # Initialize the trainer, specify the vocabulary size and special tokens
        self.trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.set_pre_tokenizer(pre_tokenizer=pre_tokenizer)
