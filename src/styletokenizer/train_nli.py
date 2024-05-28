from collections import Counter
from itertools import product
from datasets import load_dataset

from huggingface_tokenizers import ALL_TOKENIZERS
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


def do_train(tokenizer_func: str = None, features: str = "common_words"):
    assert features in ["common_words", "cross_words", None]
    common_words = False
    cross_words = False
    if features == "common_words":
        common_words = True
        cross_words = False
    elif features == "cross_words":
        cross_words = True
        common_words = False

    # Load the SNLI dataset
    #   0 - entailment
    #   1 - neutral
    #   2 - contradiction
    snli_dataset = load_dataset('snli')
    train_dataset = snli_dataset['train'].filter(lambda x: x['label'] != -1)
    test_dataset = snli_dataset['test'].filter(lambda x: x['label'] != -1)

    import pandas as pd

    # Convert to pandas DataFrame for easier manipulation
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(test_dataset)

    # test_df = pd.DataFrame(snli_dataset['test'])

    # Function to preprocess the data (concatenate premise and hypothesis)
    def preprocess(dataframe):
        dataframe = dataframe.dropna(subset=['premise', 'hypothesis', 'label'])
        dataframe['text'] = dataframe['premise'] + " " + dataframe['hypothesis']
        dataframe = dataframe[['text', 'label']]
        return dataframe

    def cross_words_preprocess(dataframe, binary=False):
        dataframe = dataframe.dropna(subset=['premise', 'hypothesis', 'label'])
        dataframe['premise_tokens'] = dataframe['premise'].apply(tokenizer_func)
        dataframe['hypothesis_tokens'] = dataframe['hypothesis'].apply(tokenizer_func)
        def counter_to_string(counter):
            return ' '.join([worda + "_" + wordb for (worda, wordb), count in counter.items() for _ in range(count)])

        dataframe['text'] = dataframe.apply(
            lambda row: counter_to_string(word_cross_product_phi(row['premise_tokens'], row['hypothesis_tokens'])),
            axis=1)

        return dataframe[['text', 'label']]

    if (not common_words) and (not cross_words):
        train_df = preprocess(train_df)
        val_df = preprocess(val_df)
    # elif common_words:
    #     train_df = common_words_preprocess(train_df)
    #     val_df = common_words_preprocess(val_df)
    elif cross_words:
        train_df = cross_words_preprocess(train_df)
        val_df = cross_words_preprocess(val_df)

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    # Initialize the TF-IDF vectorizer
    vectorizer = CountVectorizer()  # max_features=10000,

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

    # Print the classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))


for tok_name, tok_func in zip(ALL_TOKENIZERS, ALL_TOKENIZER_FUNCS):
    print(f"------- Tokenizer: {tok_name} -------")
    do_train(tok_func, features="cross_words")
