from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_name_or_path):
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