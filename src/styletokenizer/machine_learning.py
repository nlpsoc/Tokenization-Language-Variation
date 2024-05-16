import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class TextClassifier:
    def __init__(self, model, tokenizer=None, ngram=None):
        if ngram:
            self.vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True, ngram_range=(1, ngram), lowercase=False)
        else:
            self.vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True, lowercase=False)
        self.model = model

    def fit_vectorizer(self, texts):
        print("Fitting vectorizer...")
        self.vectorizer.fit(texts)
        print("Vocabulary size:", len(self.vectorizer.vocabulary_))

    def get_most_frequent_tokens(self, texts, n=100):
        X = self.vectorizer.transform(texts)
        feature_names = self.get_feature_names_out()
        token_counts = X.sum(axis=0).A1

        token_count_dict = dict(zip(feature_names, token_counts))
        sorted_tokens = sorted(token_count_dict.items(), key=lambda x: x[1], reverse=True)

        for i, (token, count) in enumerate(sorted_tokens):
            if i < n:
                print(f"{token}: {count}")

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

    def transform_texts(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

    def fit(self, texts, labels):
        print("Transforming texts to feature matrix...")
        X = self.transform_texts(texts)
        print("Feature matrix shape:", X.shape)
        self.model.fit(X, labels)

    def predict(self, texts):
        X = self.transform_texts(texts)
        return self.model.predict(X)

    def score(self, texts, true_labels):
        predicted_labels = self.predict(texts)
        return accuracy_score(true_labels, predicted_labels), predicted_labels

    def classification_report(self, texts, true_labels):
        predicted_labels = self.predict(texts)
        return classification_report(true_labels, predicted_labels)

    def confusion_matrix(self, texts, true_labels):
        predicted_labels = self.predict(texts)
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_df = pd.DataFrame(cm, index=['True Negative', 'True Positive'],
                             columns=['Predicted Negative', 'Predicted Positive'])
        return cm_df

    def evaluate(self, texts, true_labels):
        print(self.confusion_matrix(texts, true_labels))
        print(self.classification_report(texts, true_labels))
