from unittest import TestCase
from styletokenizer.logistic_regression import TextClassifier


class TestTextClassifier(TestCase):
    def setUp(self):
        self.classifier = TextClassifier()

    def test_fit_and_predict(self):
        # Define some simple training data
        texts1 = ['Hello world', 'Goodbye world']
        texts2 = ['Hello universe', 'Goodbye universe']
        labels = [0, 1]

        # Fit the classifier
        self.classifier.fit(texts1, texts2, labels)

        # Define some simple test data
        test_text1 = 'Hello world'
        test_text2 = 'Goodbye universe'

        # Predict the label of the test data
        prediction = self.classifier.predict(test_text1, test_text2)

        # Check that the prediction is as expected
        self.assertIn(prediction[0], [0, 1])

        print(self.classifier.score(texts1, texts2, labels))
        print(self.classifier.get_coefficients())