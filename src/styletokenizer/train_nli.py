import os

from styletokenizer.utility.filesystem import on_cluster

if on_cluster():
    cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

from datasets import load_dataset

from huggingface_tokenizers import ALL_TOKENIZERS
from logistic_regression import preprocess, cross_words_preprocess, set_log_reg_features, create_featurized_dataset, \
    uncommon_whitespace_tokenizer
from utility.torchtokenizer import ALL_TOKENIZER_FUNCS
from whitespace_consts import common_ws_tokenize, common_apostrophe_tokenize


def do_train(tokenizer_func: str = None, features: str = "common_words"):
    # Load the SNLI dataset
    #   0 - entailment
    #   1 - neutral
    #   2 - contradiction
    snli_dataset = load_dataset('snli')
    train_dataset = snli_dataset['train'].filter(lambda x: x['label'] != -1)
    test_dataset = snli_dataset['test'].filter(lambda x: x['label'] != -1)
    # test_df = pd.DataFrame(snli_dataset['test'])

    import pandas as pd

    # Convert to pandas DataFrame for easier manipulation
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(test_dataset)

    # Featurize the data, i.e., save texts as strings with the features we want to use
    #   ATTENTION: needs the correct whitespace tokenizer in the count vectorizer later
    train_df = create_featurized_dataset(features, tokenizer_func, train_df, text1_name="premise",
                                         text2_name="hypothesis")
    val_df = create_featurized_dataset(features, tokenizer_func, val_df, text1_name="premise", text2_name="hypothesis")

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
    # model = LogisticRegression(max_iter=1000, penalty="l2", C=0.4, solver="liblinear")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = model.predict(X_val)

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))


print(f"------ Whitespace Tokenizer ------")
do_train(common_ws_tokenize, features="cross_words")

print(f"------ Apostrophe Tokenizer ------")
do_train(common_apostrophe_tokenize, features="cross_words")

for tok_name, tok_func in zip(ALL_TOKENIZERS, ALL_TOKENIZER_FUNCS):
    print(f"------- Tokenizer: {tok_name} -------")
    do_train(tok_func, features="cross_words")
