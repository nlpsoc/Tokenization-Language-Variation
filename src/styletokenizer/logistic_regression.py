from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd


class TextClassifier:
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