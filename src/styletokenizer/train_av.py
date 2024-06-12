"""
    original version generated with GitHub Copilot April 22nd 2024
"""
import argparse
import os
import sys

sys.path.append(".")
sys.path.append("styletokenizer")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from styletokenizer.utility.filesystem import on_cluster

if on_cluster():
    cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

from huggingface_tokenizers import HUGGINGFACE_TOKENIZERS, TRAINED_TOKENIZERS
from logistic_regression import uncommon_whitespace_tokenizer, \
    create_featurized_dataset, lowercase_stem_and_clean, print_extreme_coefficients
from styletokenizer.load_data import load_pickle_file
from styletokenizer.utility.filesystem import get_dir_to_src, on_cluster

from utility.torchtokenizer import ALL_TOKENIZER_FUNCS, TorchTokenizer
from utility.umich_av import get_1_train_pairs, get_1_dev_pairs
from sklearn.model_selection import train_test_split

from whitespace_consts import common_ws_tokenize, common_apostrophe_tokenize
import pandas as pd


def main(tok_func, features: str = "common_words", reddit=False, tok_name="", load=False,
         df_dev=None, df_train=None, sample_size=1):
    if reddit:
        dev_name = f"{tok_name}_df_dev_{features}_AV-reddit.tsv"
        train_name = f"{tok_name}_df_train_{features}_AV-reddit.tsv"
    else:
        dev_name = f"{tok_name.split('/')[-1]}_df_dev_{features}_AV-{sample_size}.tsv"
        train_name = f"{tok_name.split('/')[-1]}_df_train_{features}_AV-{sample_size}.tsv"
        print("Saving to: ", dev_name, train_name)
        # check if file exists
        if not os.path.exists(train_name):
            # try saving a temporary file to check if the directory exists
            try:
                with open(train_name, "w") as f:
                    f.write("test")
            except FileNotFoundError:
                print("Directory does not exist. Creating directory")
                os.makedirs(train_name.split("/")[0])
                os.makedirs(dev_name.split("/")[0])
            os.remove(train_name)

    # try loading
    if load:
        try:
            assert load is True
            df_dev = pd.read_csv(dev_name, sep="\t")
            df_train = pd.read_csv(train_name, sep="\t")
            print("Loaded existing dataframes")
        except FileNotFoundError:
            print("Creating new dataframes")
        except AssertionError:
            print("Creating new dataframes")

    else:
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

        elif df_dev is None or df_train is None:
            df_dev, df_train = get_AV_data(sample_size)

        print("Train size: ", len(df_train))
        df_train = create_featurized_dataset(features, tok_func, df_train, symmetric=True)
        df_dev = create_featurized_dataset(features, tok_func, df_dev, symmetric=True, return_full_tokens=True)

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

    # add prediction to df_dev dataframe
    df_dev["prediction"] = y_val_pred

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_dev, y_val_pred))

    # print most predicitve features
    print_extreme_coefficients(model, vectorizer)

    # performance per source
    for source in df_dev["source"].unique():
        source_df = df_dev[df_dev["source"] == source]
        y_val_pred = model.predict(vectorizer.transform(source_df['text'].tolist()))
        print(f"Source: {source}")
        print("Classification Report:")
        print(classification_report(source_df["label"], y_val_pred))

    # save the dataframe
    df_dev.to_csv(dev_name, index=False, sep="\t")
    df_train.to_csv(train_name, index=False, sep="\t")

    print("Saved dataframes to: ", dev_name, train_name)


def get_AV_data(preprocess=False, sample_size=1, ood_dev=False):
    pairs, labels, sources = get_1_train_pairs()
    if ood_dev:
        dev_pairs, dev_labels = get_1_dev_pairs()
        train_pairs, train_labels = pairs, labels
        train_text1, train_text2 = train_pairs[0], train_pairs[1]
        dev_text1, dev_text2 = dev_pairs[0], dev_pairs[1]
    else:
        # create random split
        pairs = (pairs[0], pairs[1], labels, sources)
        # shuffle pairs
        pairs = list(zip(*pairs))
        import random
        random.seed(42)
        random.shuffle(pairs)
        pairs = pairs[:int(len(pairs) * sample_size)]
        train_pairs, dev_pairs = train_test_split(pairs, test_size=0.2, random_state=42, shuffle=True)
        train_labels = [item[2] for item in train_pairs]
        dev_labels = [item[2] for item in dev_pairs]
        train_text1, train_text2 = list(zip(*train_pairs))[0], list(zip(*train_pairs))[1]
        dev_text1, dev_text2 = list(zip(*dev_pairs))[0], list(zip(*dev_pairs))[1]
        train_sources = [item[3] for item in train_pairs]
        dev_sources = [item[3] for item in dev_pairs]
    # limit texts to 600 characters
    train_text1 = [text[:600] for text in train_text1]
    train_text2 = [text[:600] for text in train_text2]
    dev_text1 = [text[:600] for text in dev_text1]
    dev_text2 = [text[:600] for text in dev_text2]
    # create dataframe
    df_train = pd.DataFrame({"text1": train_text1, "text2": train_text2, "label": train_labels, "source": train_sources})
    df_dev = pd.DataFrame({"text1": dev_text1, "text2": dev_text2, "label": dev_labels, "source": dev_sources})

    if preprocess:
        df_dev = df_dev.apply(lambda x: lowercase_stem_and_clean(x, "text1", "text2"), axis=1)
        df_train = df_train.apply(lambda x: lowercase_stem_and_clean(x, "text1", "text2"), axis=1)
    return df_dev, df_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextClassifier model.')
    args = parser.parse_args()

    df_dev, df_train = get_AV_data(ood_dev=False, preprocess=False)

    # print(f"------ Whitespace Tokenizer ------")
    # main(reddit=False, tok_func=common_ws_tokenize, features="common_words", tok_name="whitespace")
    #
    # print(f"------ Apostrophe Tokenizer ------")
    # main(reddit=False, tok_func=common_apostrophe_tokenize, features="common_words", tok_name="apostrophe")
    #
    # print(f"------ split(" ") tokenizer ------")
    # main(reddit=False, tok_func=lambda x: x.split(" "), features="common_words", tok_name="split")

    for tok_name, tok_func in zip(HUGGINGFACE_TOKENIZERS, ALL_TOKENIZER_FUNCS):
        print(f"------- Tokenizer: {tok_name} -------")
        main(reddit=False, tok_func=tok_func, features="common_words", tok_name=tok_name, load=False,
             df_dev=df_dev, df_train=df_train)

    # for tok_name, tok_func in zip(TRAINED_TOKENIZERS, [TorchTokenizer(tok_name).tokenize for tok_name in TRAINED_TOKENIZERS]):
    #     print(f"------- Tokenizer: {tok_name} -------")
    #     main(reddit=False, tok_func=tok_func, features="common_words", tok_name=tok_name, load=False,
    #          df_dev=df_dev, df_train=df_train)
