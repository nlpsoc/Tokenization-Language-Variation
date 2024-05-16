import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from styletokenizer.machine_learning import TextClassifier


class RandomForestTextClassifier(TextClassifier):
    def __init__(self, tokenizer=None, ngram=None):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        super().__init__(model, tokenizer, ngram)


class RandomForestTextPairClassifier(TextClassifier):
    def __init__(self, tokenizer=None, ngram=None):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        super().__init__(model, tokenizer, ngram)

    def fit_vectorizer(self, query_texts, candidate_texts):
        combined_texts = query_texts + candidate_texts
        super().fit_vectorizer(combined_texts)

    def transform_texts(self, query_texts, candidate_texts):
        query_bow = self.vectorizer.transform(query_texts)
        candidate_bow = self.vectorizer.transform(candidate_texts)
        combined_features = np.hstack((query_bow.toarray(), candidate_bow.toarray()))
        return combined_features

    def fit(self, query_texts, candidate_texts, labels):
        X = self.transform_texts(query_texts, candidate_texts)
        super().fit(X, labels)

    def predict(self, query_texts, candidate_texts):
        X = self.transform_texts(query_texts, candidate_texts)
        return self.model.predict(X)

    def score(self, query_texts, candidate_texts, true_labels):
        X = self.transform_texts(query_texts, candidate_texts)
        predicted_labels = self.model.predict(X)
        return accuracy_score(true_labels, predicted_labels), predicted_labels


# Example usage:
query_texts = ["query text 1", "query text 2", "query text 3"]
candidate_texts = ["candidate text 1", "candidate text 2", "candidate text 3"]
labels = [1, 0, 1]

# Random Forest Pair Classifier
rf_pair_classifier = RandomForestTextPairClassifier()
rf_pair_classifier.fit_vectorizer(query_texts, candidate_texts)
rf_pair_classifier.fit(query_texts, candidate_texts, labels)
accuracy, predictions = rf_pair_classifier.score(query_texts, candidate_texts, labels)
print(f"Random Forest Pair Classifier Accuracy: {accuracy}")