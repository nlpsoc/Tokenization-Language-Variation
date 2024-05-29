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
from utility.umich_av import get_1_train_pairs
from sklearn.model_selection import train_test_split

from whitespace_consts import common_ws_tokenize


def whitespace_tokenizer(text):
    return text.split("\u3000")

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


def main(tok_func, features: str = "common_words", reddit=False):
    assert features in ["common_words", "cross_words", None]
    common_words = False
    cross_words = False
    if features == "common_words":
        common_words = True
        cross_words = False
    elif features == "cross_words":
        cross_words = True
        common_words = False
        print("Cross words")

    if reddit:
        # Load the training data
        train_path = (get_dir_to_src() + "/../data/development/train_reddit-corpus-small-30000_dataset"
                                                    ".pickle")
        train_data = load_pickle_file(train_path)
        # dev_path = get_dir_to_src() + "/../data/development/dev_reddit-corpus-small-30000_dataset.pickle"
        # dev_data = load_pickle_file(dev_path)

        # create random split: TODO: potentially change later to a more sophisticated split
        train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)

        # Extract the texts and labels from the training data
        train_text1 = [item['u1'] for item in train_data]
        train_text2 = [item['u2'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        dev_text1 = [item['u1'] for item in dev_data]
        dev_text2 = [item['u2'] for item in dev_data]
        dev_labels = [item['label'] for item in dev_data]

    else:
        pairs, labels = get_1_train_pairs()
        # create random split
        pairs = (pairs[0], pairs[1], labels)
        # shuffle pairs
        pairs = list(zip(*pairs))
        import random
        random.seed(42)
        random.shuffle(pairs)
        # sample 10% of the data
        pairs = pairs[:int(len(pairs) * 0.1)]
        train_pairs, dev_pairs = train_test_split(pairs, test_size=0.2, random_state=42, shuffle=True)
        train_labels = [item[2] for item in train_pairs]
        dev_labels = [item[2] for item in dev_pairs]
        train_text1, train_text2 = list(zip(*train_pairs))[0], list(zip(*train_pairs))[1]
        dev_text1, dev_text2 = list(zip(*dev_pairs))[0], list(zip(*dev_pairs))[1]


    # create dataframe
    import pandas as pd
    df_train = pd.DataFrame({"text1": train_text1, "text2": train_text2, "label": train_labels})
    df_dev = pd.DataFrame({"text1": dev_text1, "text2": dev_text2, "label": dev_labels})

    def preprocess(dataframe):
        dataframe = dataframe.dropna(subset=['text1', 'text2', 'label'])
        dataframe['text'] = dataframe['text1'] + " " + dataframe['text2']
        dataframe = dataframe[['text', 'label']]
        return dataframe

    def cross_words_preprocess(dataframe, common_only=False):
        dataframe = dataframe.dropna(subset=['text1', 'text2', 'label'])
        dataframe['text1_tokens'] = dataframe['text1'].apply(tok_func)
        dataframe['text2_tokens'] = dataframe['text2'].apply(tok_func)

        def counter_to_string(counter):
            return '\u3000'.join([worda + "_" + wordb for (worda, wordb), count in counter.items() for _ in range(count)
                                  if (not common_only) or (worda == wordb)])

        dataframe['text'] = dataframe.apply(
            lambda row: counter_to_string(word_cross_product_phi(row['text1_tokens'], row['text2_tokens'])),
            axis=1)

        return dataframe[['text', 'label']]

    print("Train size: ", len(df_train))

    if (not common_words) and (not cross_words):
        df_train = preprocess(df_train)
        df_dev = preprocess(df_dev)
    elif cross_words:
        print("Applying cross words...")
        df_train = cross_words_preprocess(df_train, common_only=False)
        df_dev = cross_words_preprocess(df_dev, common_only=False)
        print("Cross words applied.")
    elif common_words:
        print("Applying common words...")
        df_train = cross_words_preprocess(df_train, common_only=True)
        df_dev = cross_words_preprocess(df_dev, common_only=True)
        print("Common words applied.")


    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(tokenizer=whitespace_tokenizer, lowercase=False)  # add tokenizer=tok_func to preprocess text
    print("Fitting BoW vectorizer")
    concatenated_texts = df_train['text'].tolist() + df_dev['text'].tolist()
    vectorizer.fit_transform(concatenated_texts)
    X_train = vectorizer.transform(df_train['text'].tolist())
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")
    y_train = df_train['label']
    X_dev = vectorizer.transform(df_dev['text'].tolist())
    y_dev = df_dev['label']

    print("Fitting Logistic Regression model")
    model = LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver="liblinear")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_dev)

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_dev, y_val_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextClassifier model.')
    args = parser.parse_args()

    print(f"------ Whitespace Tokenizer ------")
    main(reddit=False, tok_func=common_ws_tokenize, features="cross_words")

    for tok_name, tok_func in zip(ALL_TOKENIZERS, ALL_TOKENIZER_FUNCS):
        print(f"------- Tokenizer: {tok_name} -------")
        main(reddit=False, tok_func=tok_func, features="common_words")
