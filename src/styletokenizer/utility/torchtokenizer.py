from transformers import AutoTokenizer


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
        if type(text) == list:
            return [self.tokenizer.tokenize(t) for t in text]
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text)


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
