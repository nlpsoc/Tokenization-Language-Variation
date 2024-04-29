from utility import seed
seed.set_global()

from styletokenizer.logistic_regression import TextClassifier
from styletokenizer.tokenizer import Tokenizer
from styletokenizer.utility import gyafc
import random
from utility.gyafc import to_classification_data
from styletokenizer import POS

def shuffle_lists_in_unison(list1, list2):
    """ generated CoPilot, April 29 2024

    """
    # Create a list of indices
    indices = list(range(len(list1)))
    # Shuffle the indices
    random.shuffle(indices)
    # Use the shuffled indices to rearrange both lists
    list1 = [list1[i] for i in indices]
    list2 = [list2[i] for i in indices]
    return list1, list2


def main(tokenizer_name=None, ngram=None):
    # Load the training data
    train_data = gyafc.load_train_data()
    train_labels, train_texts = to_classification_data(train_data)
    #   shuffle both lists the same way
    train_texts, train_labels = shuffle_lists_in_unison(train_texts, train_labels)

    # Set the Tokenizer
    print(f"Setting tokenizer to {tokenizer_name}")
    if not tokenizer_name:
        classifier = TextClassifier(ngram=ngram)
    elif tokenizer_name == "POS":
        classifier = TextClassifier(tokenizer=POS.tag, ngram=ngram)
    elif tokenizer_name:
        tokenizer = Tokenizer(tokenizer_name)
        classifier = TextClassifier(tokenizer=tokenizer.tokenize, ngram=ngram)

    classifier.fit_vectorizer(train_texts)

    # Fit the TextClassifier model on the training data
    classifier.fit(train_texts, train_labels)

    # Load the development data
    dev_data = gyafc.load_dev_data()
    dev_labels, dev_texts = to_classification_data(dev_data)

    # Evaluate the model on the development data
    score = classifier.score(dev_texts, dev_labels)

    # Print the performance of the model
    print(f'Model performance on development set: {score}')
    print(f'Number of features: {len(classifier.get_coefficients())}')
    print(f'Formal set to 1 in classification')
    classifier.print_extreme_coefficients()


main(tokenizer_name="POS", ngram=3)
