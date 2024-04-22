from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse


class TextClassifier:
    """ generated with GitHub Copilot April 22nd 2024
    A binary classifier using Logistic Regression on two sets of texts.

    Both sets of texts are converted to a bag of words using the same CountVectorizer, resulting in two feature matrices.
    These matrices are concatenated horizontally to form the final feature matrix.

    The fit method takes two lists of texts and a corresponding list of labels, and fits the Logistic Regression model.
    The predict method takes two texts, converts each to a separate bag of words, concatenates the feature vectors,
    and uses the Logistic Regression model to make a prediction.
    """
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()

    def fit(self, texts1, texts2, labels):
        # Combine the two sets of texts
        combined_texts = texts1 + texts2
        # Fit the CountVectorizer on the combined texts
        self.vectorizer.fit(combined_texts)
        # Convert each set of texts to a bag of words
        X1 = self.vectorizer.transform(texts1)
        X2 = self.vectorizer.transform(texts2)
        # Concatenate the two feature matrices horizontally
        X = scipy.sparse.hstack([X1, X2])
        # Use the provided labels
        y = labels
        # Fit the Logistic Regression model
        self.model.fit(X, y)

    def predict(self, text1, text2):
        # Convert each text to a bag of words
        X1 = self.vectorizer.transform([text1])
        X2 = self.vectorizer.transform([text2])
        # Concatenate the two feature vectors horizontally
        X = scipy.sparse.hstack([X1, X2])
        # Use the Logistic Regression model to make a prediction
        return self.model.predict(X)

    def score(self, texts1, texts2, labels):
        # Convert each set of texts to a bag of words
        X1 = self.vectorizer.transform(texts1)
        X2 = self.vectorizer.transform(texts2)
        # Concatenate the two feature matrices horizontally
        X = scipy.sparse.hstack([X1, X2])
        # Use the Logistic Regression model to score the prediction (accuracy)
        return self.model.score(X, labels)

    def get_coefficients(self):
        # Get the feature names from the CountVectorizer
        feature_names = self.vectorizer.get_feature_names_out()
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