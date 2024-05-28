"""
    original version generated with GitHub Copilot April 22nd 2024
"""
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from huggingface_tokenizers import ALL_TOKENIZERS
from styletokenizer.load_data import load_pickle_file
from styletokenizer.logistic_regression import TextsClassifier
from styletokenizer.utility.filesystem import get_dir_to_src
from styletokenizer.tokenizer import TorchTokenizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
from itertools import product
from datasets import load_dataset

from utility.torchtokenizer import ALL_TOKENIZER_FUNCS


def word_cross_product_phi(t1, t2):
    """Basis for cross-product features. This tends to produce pretty
    dense representations.

    Parameters
    ----------
    t1, t2 : list of str
        Tokenized premise and hypothesis.

    Returns
    -------
    Counter
        Maps each (w1, w2) in the cross-product of `t1` and `t2` to its count.
    """
    return Counter([(w1, w2) for w1, w2 in product(t1, t2)])


def main(train_path, dev_path, tok_func, features: str = "common_words"):
    assert features in ["common_words", "cross_words", None]
    common_words = False
    cross_words = False
    if features == "common_words":
        common_words = True
        cross_words = False
    elif features == "cross_words":
        cross_words = True
        common_words = False

    # Load the training data
    train_data = load_pickle_file(train_path)
    # Load the development data
    # dev_data = load_pickle_file(dev_path)

    # create random split
    from sklearn.model_selection import train_test_split
    train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Extract the texts and labels from the training data
    train_text1 = [item['u1'] for item in train_data]
    train_text2 = [item['u2'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    dev_text1 = [item['u1'] for item in dev_data]
    dev_text2 = [item['u2'] for item in dev_data]
    dev_labels = [item['label'] for item in dev_data]

    # create dataframe
    import pandas as pd
    df_train = pd.DataFrame({"text1": train_text1, "text2": train_text2, "label": train_labels})
    df_dev = pd.DataFrame({"text1": dev_text1, "text2": dev_text2, "label": dev_labels})

    def preprocess(dataframe):
        dataframe = dataframe.dropna(subset=['text1', 'text2', 'label'])
        dataframe['text'] = dataframe['text1'] + " " + dataframe['text2']
        dataframe = dataframe[['text', 'label']]
        return dataframe

    def cross_words_preprocess(dataframe, binary=False):
        dataframe = dataframe.dropna(subset=['text1', 'text2', 'label'])
        dataframe['text1_tokens'] = dataframe['text1'].apply(tok_func)
        dataframe['text2_tokens'] = dataframe['text2'].apply(tok_func)

        def counter_to_string(counter):
            return ' '.join([worda + "_" + wordb for (worda, wordb), count in counter.items() for _ in range(count)])

        dataframe['text'] = dataframe.apply(
            lambda row: counter_to_string(word_cross_product_phi(row['text1'], row['text2'])),
            axis=1)

        return dataframe[['text', 'label']]

    if (not common_words) and (not cross_words):
        df_train = preprocess(df_train)
        df_dev = preprocess(df_dev)
    elif cross_words:
        df_train = cross_words_preprocess(df_train)
        df_dev = cross_words_preprocess(df_dev)

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    vectorizer = CountVectorizer()  # add tokenizer=tok_func to preprocess text
    X_train = vectorizer.fit_transform(df_train['text'])
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")
    y_train = df_train['label']
    X_dev = vectorizer.transform(df_dev['text'])
    y_dev = df_dev['label']

    model = LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver="liblinear")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_dev)

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_dev, y_val_pred))


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

    for tok_name, tok_func in zip(ALL_TOKENIZERS, ALL_TOKENIZER_FUNCS):
        print(f"------- Tokenizer: {tok_name} -------")
        main(train_path=args.train_path, dev_path=args.dev_path, tok_func=tok_func, features="cross_words")
