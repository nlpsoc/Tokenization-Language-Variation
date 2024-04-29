from unittest import TestCase
from styletokenizer.logistic_regression import TextsClassifier


class TestTextClassifier(TestCase):
    def setUp(self):
        self.classifier = TextsClassifier()

    def test_fit_and_predict(self):
        # Define some simple training data
        texts1 = ['don’t suggest an open relationship if you’re not ready', 'don’t suggest an open relationship if you’re not ready']
        texts2 = ['it’s clear that these are wildly different situations', 'Aren\'t open relationships usually just about fixing something in the relationship?']
        labels = [1, 0]

        # Fit the classifier
        self.classifier.fit(texts1, texts2, labels)

        # Define some simple test data
        test_text1 = 'don’t suggest an open relationship if you’re not ready'
        test_text2 = 'it’s clear that these are wildly different situations'

        # Predict the label of the test data
        prediction = self.classifier.predict(test_text1, test_text2)

        # Check that the prediction is as expected
        self.assertIn(prediction[0], [0, 1])

        print(self.classifier.score(texts1, texts2, labels))
        print(self.classifier.get_coefficients())