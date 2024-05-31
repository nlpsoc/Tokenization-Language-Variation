"""
    original version generated with GitHub Copilot April 22nd 2024
"""
import argparse
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from styletokenizer.utility.filesystem import on_cluster

if on_cluster():
    cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

from huggingface_tokenizers import ALL_TOKENIZERS
from logistic_regression import uncommon_whitespace_tokenizer, \
    create_featurized_dataset
from styletokenizer.load_data import load_pickle_file
from styletokenizer.utility.filesystem import get_dir_to_src, on_cluster

from utility.torchtokenizer import ALL_TOKENIZER_FUNCS
from utility.umich_av import get_1_train_pairs
from sklearn.model_selection import train_test_split

from whitespace_consts import common_ws_tokenize, common_apostrophe_tokenize


def main(tok_func, features: str = "common_words", reddit=False):


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
        # pote
        sample_size = 1
        pairs = pairs[:int(len(pairs) * sample_size)]
        train_pairs, dev_pairs = train_test_split(pairs, test_size=0.2, random_state=42, shuffle=True)
        train_labels = [item[2] for item in train_pairs]
        dev_labels = [item[2] for item in dev_pairs]
        train_text1, train_text2 = list(zip(*train_pairs))[0], list(zip(*train_pairs))[1]
        dev_text1, dev_text2 = list(zip(*dev_pairs))[0], list(zip(*dev_pairs))[1]

    # create dataframe
    import pandas as pd
    df_train = pd.DataFrame({"text1": train_text1, "text2": train_text2, "label": train_labels})
    df_dev = pd.DataFrame({"text1": dev_text1, "text2": dev_text2, "label": dev_labels})

    print("Train size: ", len(df_train))

    df_train = create_featurized_dataset(features, tok_func, df_train, symmetric=True)
    df_dev = create_featurized_dataset(features, tok_func, df_dev, symmetric=True)

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(tokenizer=uncommon_whitespace_tokenizer,
                                 lowercase=False)  # add tokenizer=tok_func to preprocess text
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

    # print(f"------ Whitespace Tokenizer ------")
    # main(reddit=False, tok_func=common_ws_tokenize, features="cross_words")
    #
    # print(f"------ Apostrophe Tokenizer ------")
    # main(reddit=False, tok_func=common_apostrophe_tokenize, features="cross_words")

    for tok_name, tok_func in zip(ALL_TOKENIZERS, ALL_TOKENIZER_FUNCS):
        print(f"------- Tokenizer: {tok_name} -------")
        main(reddit=False, tok_func=tok_func, features="cross_words")
