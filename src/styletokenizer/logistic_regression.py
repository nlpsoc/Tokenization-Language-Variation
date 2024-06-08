import math
from collections import Counter
from itertools import product
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

import styletokenizer.machine_learning as machine_learning


class TextClassifier:
    def __init__(self, tokenizer=None, ngram=None):
        if ngram:
            self.vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True, ngram_range=(1, ngram),
                                              lowercase=False)
        else:
            self.vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True, lowercase=False)  # deterministic
        self.model = LogisticRegression(random_state=123, max_iter=1000)  # , penalty='l1', solver='saga', C=1.0

    def fit_vectorizer(self, texts):
        print("Fitting vectorizer...")
        self.vectorizer.fit(texts)
        print("Vocabulary size:", len(self.vectorizer.vocabulary_))

    def get_most_frequent_tokens(self, texts, n=100):
        # Assume `texts` is your list of texts and `vectorizer` is your fitted CountVectorizer
        X = self.vectorizer.transform(texts)
        # Get the feature names (tokens)
        feature_names = self.vectorizer.get_feature_names_out()
        # Sum the counts of each token across all documents
        token_counts = X.sum(axis=0).A1

        # Create a dictionary mapping tokens to their counts
        token_count_dict = dict(zip(feature_names, token_counts))

        # Sort tokens by counts in descending order
        sorted_tokens = sorted(token_count_dict.items(), key=lambda x: x[1], reverse=True)

        # Print the most frequent tokens with counts
        for i, (token, count) in enumerate(sorted_tokens):
            if i < 200:
                print(f"{token}: {count}")

        # Print the most frequent tokens with counts
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{|c|c|}")
        print("\\hline")
        print("Token & Count \\\\")
        print("\\hline")
        for i, (token, count) in enumerate(sorted_tokens[:20]):
            print(f"{token} & {count} \\\\")
            if i < 19:
                print("\\hline")
        print("\\hline")
        print("\\end{tabular}")
        print("\\caption{Top 20 Tokens and Counts}")
        print("\\label{tab:top_tokens}")
        print("\\end{table}")

    def fit(self, texts, labels):
        from utility import seed
        seed.set_global()

        print("Transforming texts to feature matrix...")
        X = self.vectorizer.transform(texts)
        print("Feature matrix shape:", X.shape)
        self.model.fit(X, labels)
        if not self.model.n_iter_ < self.model.max_iter:
            print("Model did not converge!")
        else:
            print("Model converged after", self.model.n_iter_, "iterations.")

    def predict(self, texts):
        # Convert texts to a bag of words
        X = self.vectorizer.transform(texts)
        # Use the Logistic Regression model to make predictions
        return self.model.predict(X)

    def score(self, texts, true_labels):
        # Convert texts to a bag of words
        X = self.vectorizer.transform(texts)
        # Use the Logistic Regression model to make predictions
        predicted_labels = self.model.predict(X)
        # Return the mean accuracy of the predictions
        return accuracy_score(true_labels, predicted_labels), predicted_labels

    def get_coefficients(self):
        # Get the feature names from the CountVectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        # Get the coefficients from the Logistic Regression model
        coefficients = self.model.coef_[0]
        # Return a dictionary mapping feature names to coefficients
        return dict(zip(feature_names, coefficients))

    def print_extreme_coefficients(self, n=100):
        # Get the coefficients
        coefficients = self.get_coefficients()

        # Sort the coefficients by value
        sorted_coefficients = sorted(coefficients.items(), key=lambda item: item[1])

        # Print the smallest coefficients
        print(f'Smallest coefficients:')
        for word, coefficient in sorted_coefficients[:n]:
            print(f'{word}: {coefficient}')

        # Print the largest coefficients
        print(f'\nLargest coefficients:')
        for word, coefficient in sorted_coefficients[-n:]:
            print(f'{word}: {coefficient}')

        # Print a few examples where the coefficient is 0
        print(f'\nCoefficients close to 0:')
        zero_coefficients = [item for item in sorted_coefficients if math.isclose(item[1], 0, abs_tol=0.001)]
        for word, coefficient in zero_coefficients[:n]:
            print(f'{word}: {coefficient}')


class TextsClassifier:
    """ originally generated with GitHub Copilot April 22nd 2024
    A binary classifier using Logistic Regression on two sets of texts.

    Both sets of texts are converted to a bag of words using the same CountVectorizer, resulting in two feature matrices.
    These matrices are concatenated horizontally to form the final feature matrix.

    The fit method takes two lists of texts and a corresponding list of labels, and fits the Logistic Regression model.
    The predict method takes two texts, converts each to a separate bag of words, concatenates the feature vectors,
    and uses the Logistic Regression model to make a prediction.
    """

    def __init__(self, tokenizer=None):
        self.vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
        self.model = LogisticRegression(max_iter=1000, tol=1e-4, penalty='l1')
        # self.poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
        self.interaction_columns = []

    def _get_interaction_features(self, X1, X2):
        # Compute interaction features only between different texts
        # Outer product between text1 and text2 (cross-text interactions)
        interaction_features = [np.outer(t1, t2).flatten() for t1, t2 in zip(X1, X2)]
        # Convert to DataFrame
        interaction_df = pd.DataFrame(interaction_features, columns=[
            f"{w1}_{w2}" for w1 in self.vectorizer.get_feature_names_out()
            for w2 in self.vectorizer.get_feature_names_out()
        ])
        # only return the self.interaction_columns dataframe
        for col in self.interaction_columns:
            words = col.split("_")
            other_column = f"{words[1]}_{words[0]}"
            if other_column != col:
                assert (other_column not in self.interaction_columns)
                interaction_df[col] = interaction_df[[col, other_column]].max(axis=1)
        interaction_df = interaction_df[self.interaction_columns]
        return interaction_df

    def fit(self, texts1, texts2, labels):
        # Combine the two sets of texts
        combined_texts = texts1 + texts2
        self._fit_bow(combined_texts)
        # Convert each set of texts to a bag of words
        X1 = self.vectorizer.transform(texts1).toarray()
        X2 = self.vectorizer.transform(texts2).toarray()
        interaction_df = self._get_interaction_features(X1, X2)
        # Use the provided labels
        y = labels
        # Fit the Logistic Regression model
        self.model.fit(interaction_df, y)

    def _fit_bow(self, combined_texts):
        # Fit the CountVectorizer on the combined texts, get the interaction columns
        self.vectorizer.fit(combined_texts)
        vocab = self.vectorizer.get_feature_names_out()
        print(f"{len(vocab)} words in vocabulary")
        self.interaction_columns = [f"{v1}_{v2}" for i1, v1 in enumerate(vocab) for v2 in vocab[range(i1, len(vocab))]]
        print(f"{len(self.interaction_columns)} interaction features")

    def predict(self, text1, text2):
        # Convert each text to a bag of words
        X1 = self.vectorizer.transform([text1]).toarray()
        X2 = self.vectorizer.transform([text2]).toarray()
        interaction_df = self._get_interaction_features(X1, X2)
        # Concatenate the two feature vectors horizontally
        # X = scipy.sparse.hstack([X1, X2])
        # Use the Logistic Regression model to make a prediction
        return self.model.predict(interaction_df)

    def score(self, texts1, texts2, labels):
        # Convert each set of texts to a bag of words
        X1 = self.vectorizer.transform(texts1).toarray()
        X2 = self.vectorizer.transform(texts2).toarray()
        interaction_df = self._get_interaction_features(X1, X2)
        # Concatenate the two feature matrices horizontally
        # X = scipy.sparse.hstack([X1, X2])
        # Use the Logistic Regression model to score the prediction (accuracy)
        return self.model.score(interaction_df, labels)

    def get_coefficients(self):
        # Get the feature names from the CountVectorizer
        feature_names = self.interaction_columns
        # Get the coefficients from the Logistic Regression model
        coefficients = self.model.coef_[0]
        # Return a dictionary mapping feature names to coefficients
        return dict(zip(feature_names, coefficients))

    def print_extreme_coefficients(self, n=5):
        # Get the coefficients
        coefficients = self.get_coefficients()

        # Sort the coefficients by value
        sorted_coefficients = sorted(coefficients.items(), key=lambda item: item[1])

        # Print the smallest coefficients
        print(f'Smallest coefficients:')
        for word, coefficient in sorted_coefficients[:n]:
            print(f'{word}: {coefficient}')

        # Print the largest coefficients
        print(f'\nLargest coefficients:')
        for word, coefficient in sorted_coefficients[-n:]:
            print(f'{word}: {coefficient}')

        # Print a few examples where the coefficient is 0
        print(f'\nCoefficients equal to 0:')
        zero_coefficients = [item for item in sorted_coefficients if item[1] == 0]
        for word, coefficient in zero_coefficients[:n]:
            print(f'{word}: {coefficient}')


def preprocess(dataframe, text1_name="text1", text2_name="text2", label_name="label"):
    dataframe = dataframe.dropna(subset=[text1_name, text2_name, label_name])
    dataframe['text'] = dataframe[text1_name] + " " + dataframe[text2_name]
    dataframe = dataframe[['text', label_name]]
    return dataframe


def cross_words_preprocess(tok_func, dataframe, common_only=False, uncommon_only=False, text1_name="text1", text2_name="text2",
                           label_name="label", symmetric=False, return_full_tokens=False):
    # make sure common only and uncommon only are not both true
    assert not (common_only and uncommon_only)
    if uncommon_only:
        symmetric = True
    dataframe = dataframe.dropna(subset=[text1_name, text2_name, label_name])
    # cut of texts at word length of 130
    dataframe[text1_name] = dataframe[text1_name].apply(lambda x: " ".join(x.split()[:]))  # :100
    dataframe[text2_name] = dataframe[text2_name].apply(lambda x: " ".join(x.split()[:]))  # :100

    dataframe['text1_tokens'] = dataframe[text1_name].apply(tok_func)
    dataframe['text2_tokens'] = dataframe[text2_name].apply(tok_func)

    # get number of unique tokens
    unique_tokens = set()
    for tokens in dataframe['text1_tokens']:
        unique_tokens.update(tokens)
    for tokens in dataframe['text2_tokens']:
        unique_tokens.update(tokens)
    print("Number of unique tokens:", len(unique_tokens))

    # GET STATS
    #   number of texts
    print("Number of texts:", len(dataframe))
    #   average number of words per text1
    dataframe['text1_num_words'] = dataframe[text1_name].apply(lambda x: len(x.split()))
    print("Average number of words per text1:", dataframe['text1_num_words'].mean())
    #   average number of words per text2
    dataframe['text2_num_words'] = dataframe[text2_name].apply(lambda x: len(x.split()))
    print("Average number of words per text2:", dataframe['text2_num_words'].mean())
    print("Average number of words per text1 and text2:",
          (dataframe['text1_num_words'] + dataframe['text2_num_words']).mean())
    #   get the avg number tokens per text1
    dataframe['text1_num_tokens'] = dataframe['text1_tokens'].apply(len)
    print("Average number of tokens per text1:", dataframe['text1_num_tokens'].mean())
    #   get the avg number tokens per text2
    dataframe['text2_num_tokens'] = dataframe['text2_tokens'].apply(len)
    print("Average number of tokens per text2:", dataframe['text2_num_tokens'].mean())
    print("Total number of tokens per text1 and text2:",
          (dataframe['text1_num_tokens'] + dataframe['text2_num_tokens']).mean())
    #   average number of tokens per word
    #       Calculate the average number of tokens per word for each column
    dataframe['text1_avg_tokens_per_word'] = dataframe['text1_num_tokens'] / dataframe['text1_num_words']
    dataframe['text2_avg_tokens_per_word'] = dataframe['text2_num_tokens'] / dataframe['text2_num_words']
    #       Calculate the overall average number of tokens per word for each column
    text1_avg_tokens_per_word = dataframe['text1_avg_tokens_per_word'].mean()
    text2_avg_tokens_per_word = dataframe['text2_avg_tokens_per_word'].mean()
    print(f"Average number of tokens per word for text1: {text1_avg_tokens_per_word}")
    print(f"Average number of tokens per word for text2: {text2_avg_tokens_per_word}")
    print(
        f"Average number of tokens per word for text1 and text2: {(text1_avg_tokens_per_word + text2_avg_tokens_per_word) / 2}")

    def counter_to_string(counter):  # TODO: this is inefficient, counting features twice like this
        # return empty string if counter is empty
        if not counter:
            return ""
        if not uncommon_only:
            return '\u3000'.join([worda + "_" + wordb for (worda, wordb), count in counter.items() for _ in range(count)
                                  if (not common_only) or (worda == wordb)])
        else:
            return '\u3000'.join([word for word, count in counter.items() for _ in range(count)])

    dataframe['text'] = dataframe.apply(
        lambda row: counter_to_string(get_feature_counts(row['text1_tokens'], row['text2_tokens'],
                                                         symmetric=symmetric, uncommon_only=uncommon_only)), axis=1)

    if return_full_tokens:
        return dataframe[['text', label_name, 'text1_tokens', 'text2_tokens']]
    else:
        return dataframe[['text', label_name]]


def set_log_reg_features(features):
    """
        Set the features for logistic regression, currently only common words and cross words are supported.
        I.e., common words will only use the common words between the two texts,
            while cross words will use all combinations
    :param features:
    :return:
    """
    assert features in ["common_words", "cross_words", "uncommon_words", None]
    common_words = False
    cross_words = False
    uncommon_words = False
    if features == "common_words":
        common_words = True
        cross_words = False
    elif features == "cross_words":
        cross_words = True
        common_words = False
    elif features == "uncommon_words":
        uncommon_words = True
    return common_words, cross_words, uncommon_words


def uncommon_whitespace_tokenizer(text):
    return text.split("\u3000")


def count_unique_strings(list1: List[str], list2: List[str]) -> Dict[str, int]:
    """
        given list of strings, return the count of unique strings in list1 and list2
            only if they are unique to one list
    :param list1:
    :param list2:
    :return:
    """
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Find the symmetric difference
    unique_strings = set1.symmetric_difference(set2)

    # Count occurrences in the original lists
    count1 = Counter(list1)
    count2 = Counter(list2)

    result = {}
    # Add counts
    for string in unique_strings:
        if string in count1:
            result[string] = count1[string]
        elif string in count2:
            result[string] = count2[string]
        else:
            raise ValueError(f"Something went wrong. String {string} not in either list")
    return result


def get_feature_counts(t1: List[str], t2: List[str], symmetric=False, uncommon_only=False):
    if not uncommon_only:
        result = word_cross_product(t1, t2, symmetric)
    else:
        result = count_unique_strings(t1, t2)
    return result


def word_cross_product(t1, t2, symmetric) -> Dict[Tuple[str, str], int]:
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
        :param symmetric:
    """
    counters = Counter([(w1, w2) for w1, w2 in product(t1, t2)])

    if symmetric:
        result = Counter()
        # combine the two counters (w1, w2) and (w2, w1) to one if w1 != w2
        for (w1, w2), count in counters.items():
            word_order = (w1, w2)
            if w1 != w2 and w1 > w2:
                # decide order between w1 and w2
                word_order = (w2, w1)
            if word_order in result:
                result[word_order] += count
            else:
                result[word_order] = count
        return result
    else:
        return counters


def create_featurized_dataset(features, tok_func, df_train, text1_name="text1", text2_name="text2", symmetric=False,
                              return_full_tokens=False):
    common_words, cross_words, uncommon_words = set_log_reg_features(features)
    # similar to https://github.com/duanzhihua/cs224u-1/blob/master/nli_02_models.ipynb
    if (not common_words) and (not cross_words) and (not uncommon_words):
        df_train = preprocess(df_train)
    elif cross_words:
        print("Applying cross words...")
        df_train = cross_words_preprocess(tok_func, df_train, common_only=False,
                                          text1_name=text1_name, text2_name=text2_name, symmetric=symmetric,
                                          return_full_tokens=return_full_tokens)
        print("Cross words applied.")
    elif common_words:
        print("Applying common words...")
        df_train = cross_words_preprocess(tok_func, df_train, common_only=True,
                                          text1_name=text1_name, text2_name=text2_name,
                                          return_full_tokens=return_full_tokens)
        print("Common words applied.")
    elif uncommon_words:
        print("Applying uncommon words...")
        df_train = cross_words_preprocess(tok_func, df_train, uncommon_only=True,
                                          text1_name=text1_name, text2_name=text2_name,
                                          return_full_tokens=return_full_tokens)
        print("Uncommon words applied.")
    # check that none of the columns contain nan values
    assert not df_train.isnull().values.any()
    return df_train
