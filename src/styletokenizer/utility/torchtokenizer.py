from transformers import AutoTokenizer
from tokenizers import Tokenizer

from huggingface_tokenizers import HUGGINGFACE_TOKENIZERS


class TorchTokenizer:
    def __init__(self, model_name_or_path):
        # if "deberta" in model_name_or_path:
        #     from transformers import DebertaTokenizer
        #
        #     # Load the tokenizer
        #     self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        # elif "roberta" in model_name_or_path:
        #     from transformers import RobertaTokenizer
        #
        #     # Load the tokenizer
        #     self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # else:
        # if "wikitext" in model_name_or_path:
        #     self.tokenizer = Tokenizer.from_file(model_name_or_path)
        # else:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def tokenize(self, text):
        """
            currently doing this with the actual tokens for interpretability,
                eventually prob better to use the IDs for efficiency
            unfortunately, no support for batch tokenization as far as I see for huggingface
                --> no sense doing batch tokenization and then re-substitution IDs with tokens
                    cause that means looping over everything anyway
        :param text:
        :return:
        """
        # if type(text) == list:
        #     return [self.tokenizer.tokenize(t) for t in text]
        # return self.tokenizer.tokenize(text)
        if type(text) == list:
            return [self._single_tokenize(t) for t in text]
        return self._single_tokenize(text)

    def _single_tokenize(self, text):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def normalize(self, text):
        # check if normalizer exists
        if hasattr(self.tokenizer, "normalize"):
            return self.tokenizer.normalize(text)
        else:
            return text

    def encode(self, text):
        return self.tokenizer.encode(text)


ALL_TOKENIZER_FUNCS = [TorchTokenizer(tokenizer_name).tokenize for tokenizer_name in HUGGINGFACE_TOKENIZERS]

class XLMRobertaTokenizer(TorchTokenizer):
    def __init__(self):
        super().__init__('xlm-roberta-base')


class RobertaTokenizer(TorchTokenizer):
    def __init__(self):
        super().__init__('roberta-base')


class BertTokenizer(TorchTokenizer):
    def __init__(self):
        super().__init__('bert-base-uncased')


class BERTTokenizer(TorchTokenizer):
    def __init__(self):
        super().__init__('bert-base-cased')
