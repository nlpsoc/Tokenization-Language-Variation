from unittest import TestCase

import huggingface_tokenizers
from random_forest import RandomForestTextPairClassifier
from utility.torchtokenizer import TorchTokenizer
from utility.umich_av import get_1_dev_pairs, get_1_train_pairs


class TestRandomForestTextPairClassifier(TestCase):

    def test_small(self):
        # Example usage:
        query_texts = ["query text 1", "query text 2", "query text 3"]
        candidate_texts = ["candidate text 1", "candidate text 2", "candidate text 3"]
        labels = [1, 0, 1]

        # Random Forest Pair Classifier
        rf_pair_classifier = RandomForestTextPairClassifier()
        rf_pair_classifier.fit_vectorizer((query_texts, candidate_texts))
        rf_pair_classifier.fit((query_texts, candidate_texts), labels)

        # Evaluate
        rf_pair_classifier.evaluate((query_texts, candidate_texts), labels)
        rf_pair_classifier.visualize_tree()

    def test_train(self):
        # Load UMich AV dataset
        pairs, labels = get_1_train_pairs()
        print(len(pairs[0]))

        percentage = 0.5
        pairs = (pairs[0][:int(len(pairs[0]) * percentage)], pairs[1][:int(len(pairs[1]) * percentage)])
        labels = labels[:int(len(labels) * percentage)]

        print(f"Size of pairs: {len(pairs[0])}")

        for tokenizer_name in huggingface_tokenizers.HUGGINGFACE_TOKENIZERS:
            print(f"Tokenizer: { tokenizer_name.split('/')[-1]}")
            tokenizer = TorchTokenizer(tokenizer_name)

            # Random Forest Pair Classifier
            rf_pair_classifier = RandomForestTextPairClassifier(tokenizer=tokenizer.tokenize)
            rf_pair_classifier.fit_vectorizer(pairs)
            rf_pair_classifier.fit(pairs, labels)

            # Evaluate
            pairs, labels = get_1_dev_pairs()
            rf_pair_classifier.evaluate(pairs, labels)
            rf_pair_classifier.visualize_tree(0, tokenizer_name.split("/")[-1])




