"""
    original version generated with GitHub Copilot April 22nd 2024
"""
import argparse
from styletokenizer.load_data import load_pickle_file
from styletokenizer.logistic_regression import TextClassifier
from styletokenizer.utility.filesystem import get_dir_to_src
from styletokenizer.tokenizer import Tokenizer


def main(train_path, dev_path):
    tokenizer = Tokenizer("bert-base-uncased")
    # Load the training data
    train_data = load_pickle_file(train_path)

    # Extract the texts and labels from the training data
    train_texts1 = [item['u1'] for item in train_data]
    train_texts2 = [item['u2'] for item in train_data]
    train_labels = [item['label'] for item in train_data]

    # Initialize the TextClassifier
    classifier = TextClassifier(tokenizer=tokenizer.tokenize)

    # Fit the TextClassifier model on the training data
    classifier.fit(train_texts1, train_texts2, train_labels)

    # Load the development data
    dev_data = load_pickle_file(dev_path)

    # Extract the texts and labels from the development data
    dev_texts1 = [item['u1'] for item in dev_data]
    dev_texts2 = [item['u2'] for item in dev_data]
    dev_labels = [item['label'] for item in dev_data]

    # Evaluate the model on the development data
    score = classifier.score(dev_texts1, dev_texts2, dev_labels)

    # Print the performance
    print(f'Model performance on development set: {score}')
    print(f'Number of features: {len(classifier.get_coefficients())}')
    classifier.print_extreme_coefficients()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextClassifier model.')
    parser.add_argument('-train_path', metavar='train_path', type=str,
                        default=(get_dir_to_src() + "/../data/development/train_reddit-corpus-small-30000_dataset"
                                                    ".pickle"),
                        help='the path to the training data file')
    parser.add_argument('-dev_path', metavar='dev_path', type=str,
                        default=get_dir_to_src() + "/../data/development/dev_reddit-corpus-small-30000_dataset.pickle",
                        help='the path to the development data file')
    args = parser.parse_args()
    main(train_path=args.train_path, dev_path=args.dev_path)
