import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from styletokenizer.machine_learning import TextClassifier
from sklearn.tree import export_graphviz
import graphviz

class RandomForestTextClassifier(TextClassifier):
    def __init__(self, tokenizer=None, ngram=None):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        super().__init__(model, tokenizer, ngram)


class RandomForestTextPairClassifier(TextClassifier):
    def __init__(self, tokenizer=None, ngram=None):
        model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=10)
        super().__init__(model, tokenizer, ngram)

    def fit_vectorizer(self, input_list):
        query_texts, candidate_texts = input_list
        combined_texts = query_texts + candidate_texts
        super().fit_vectorizer(combined_texts)

    def transform_texts(self, input_list):
        query_texts, candidate_texts = input_list
        query_bow = self.vectorizer.transform(query_texts)
        candidate_bow = self.vectorizer.transform(candidate_texts)
        combined_features = np.hstack((query_bow.toarray(), candidate_bow.toarray()))
        return combined_features

    def get_feature_names_out(self):
        feature_names = self.vectorizer.get_feature_names_out()
        return np.concatenate([feature_names + "_1", feature_names + "_2"])

    def get_ascii_feature_names_out(self):
        feature_names = self.get_feature_names_out()
        return [name.encode('ascii', 'ignore').decode('ascii') for name in feature_names]

    # Function to sanitize feature names
    def sanitize_feature_names(self):
        feature_names = self.get_feature_names_out()
        sanitized_names = [re.sub(r'\W+', '_', name) for name in feature_names]
        return sanitized_names

    def feature_importances(self):
        feature_names = self.get_feature_names_out()
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def print_top_feature_importances(self, n=20):
        importances = self.feature_importances()
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)

        print(f'Top {n} feature importances:')
        for word, importance in sorted_importances[:n]:
            print(f'{word}: {importance}')

    def evaluate(self, input_list, true_labels):
        self.print_top_feature_importances()
        super().evaluate(input_list, true_labels)

    def visualize_tree(self, tree_index=0, tok_name=""):
        feature_names = self.sanitize_feature_names()
        estimator = self.model.estimators_[tree_index]
        dot_data = export_graphviz(estimator,
                                   out_file=None,
                                   feature_names=feature_names,
                                   filled=True,
                                   rounded=True,
                                   special_characters=True)
        graph = graphviz.Source(dot_data)
        # saving tree to png file
        png_bytes = graph.pipe(format='png')
        with open(tok_name + '_decision-tree-0.png', 'wb') as f:
            f.write(png_bytes)
        return graph


