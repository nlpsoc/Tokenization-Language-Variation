import os
# add current folder to path
import sys
import pandas as pd

sys.path.append(".")
sys.path.append("styletokenizer")

from styletokenizer.utility.filesystem import on_cluster

if on_cluster():
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

from datasets import load_dataset

from huggingface_tokenizers import HUGGINGFACE_TOKENIZERS, TRAINED_TOKENIZERS
from logistic_regression import create_featurized_dataset, \
    uncommon_whitespace_tokenizer, lowercase_stem_and_clean, print_extreme_coefficients
from utility.torchtokenizer import ALL_TOKENIZER_FUNCS, TorchTokenizer


# Initialize the stemmer
# Initialize the lemmatizer


# Function to map NLTK part-of-speech tags to WordNet tags


# Define a function to lowercase, remove punctuation, remove extra whitespace, and stem the text


def do_train(tokenizer_func: str = None, tok_name: str = None, features: str = "common_words", load=False,
             train_df=None, val_df=None):
    val_name = f"{tok_name.split('/')[-1]}_df_dev_{features}_NLI.tsv"
    train_name = f"{tok_name.split('/')[-1]}_df_train_{features}_NLI.tsv"
    print("Will save results to: ", val_name, train_name)

    try:
        assert load is True
        val_df = pd.read_csv(val_name, sep="\t")
        train_df = pd.read_csv(train_name, sep="\t")
        print("Loaded existing dataframes")
    except FileNotFoundError:
        train_df, val_df = create_dataframe(features, tokenizer_func, train_df, val_df)
    except AssertionError:
        train_df, val_df = create_dataframe(features, tokenizer_func, train_df, val_df)

    print("Train size: ", len(train_df))

    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the Vectorizer on the prepped data
    vectorizer = CountVectorizer(tokenizer=uncommon_whitespace_tokenizer,
                                 lowercase=False)

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_df['text'])
    # print vocab size
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")
    y_train = train_df['label']

    # Transform the validation and test data
    X_val = vectorizer.transform(val_df['text'])
    y_val = val_df['label']

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver="liblinear")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = model.predict(X_val)
    val_df["prediction"] = y_val_pred

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # print most predicitve features
    print_extreme_coefficients(model, vectorizer)

    # save the dataframe
    val_df.to_csv(val_name, index=False, sep="\t")
    train_df.to_csv(train_name, index=False, sep="\t")

    print("Saved dataframes to: ", val_name, train_name)


def create_dataframe(features, tokenizer_func, train_df, val_df):
    print("Creating new dataframes")
    # Featurize the data, i.e., save texts as strings with the features we want to use
    #   ATTENTION: needs the correct whitespace tokenizer in the count vectorizer later
    train_df = create_featurized_dataset(features, tokenizer_func, train_df, text1_name="premise",
                                         text2_name="hypothesis")
    val_df = create_featurized_dataset(features, tokenizer_func, val_df, text1_name="premise", text2_name="hypothesis")
    return train_df, val_df


def get_dataset_and_preprocess(mnli=False, preprocess=False):
    # Load the SNLI dataset
    #   0 - entailment
    #   1 - neutral
    #   2 - contradiction
    train_key = "train"
    val_key = "validation"
    if not mnli:
        snli_dataset = load_dataset('snli')
    else:
        snli_dataset = load_dataset('nyu-mll/multi_nli')
        val_key = "validation_matched"

    train_dataset = snli_dataset[train_key].filter(lambda x: x['label'] != -1)
    dev_dataset = snli_dataset[val_key].filter(lambda x: x['label'] != -1)

    if preprocess:
        train_dataset = train_dataset.map(lowercase_stem_and_clean, load_from_cache_file=False)
        dev_dataset = dev_dataset.map(lowercase_stem_and_clean, load_from_cache_file=False)

    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(dev_dataset)
    return train_df, val_df


train_df, val_df = get_dataset_and_preprocess(mnli=True, preprocess=False)
# print(f"------ split(" ") tokenizer ------")
# do_train(lambda x: x.split(" "), tok_name="split", features="cross_words")
#
# print(f"------ Whitespace Tokenizer ------")
# do_train(common_ws_tokenize, tok_name="ws", features="cross_words", load=False)
#
# print(f"------ Apostrophe Tokenizer ------")
# do_train(common_apostrophe_tokenize, tok_name="ap", features="cross_words", load=False)

# for tok_name, tok_func in zip(HUGGINGFACE_TOKENIZERS, ALL_TOKENIZER_FUNCS):
#     print(f"------- Tokenizer: {tok_name} -------")
#     do_train(tok_func, features="cross_words", tok_name=tok_name, load=False, train_df=train_df, val_df=val_df)

for tok_name, tok_func in zip(TRAINED_TOKENIZERS, [TorchTokenizer(tok_name).tokenize for tok_name in TRAINED_TOKENIZERS]):
    print(f"------- Tokenizer: {tok_name} -------")
    do_train(tok_func, features="cross_words", tok_name=tok_name, load=False, train_df=train_df, val_df=val_df)
