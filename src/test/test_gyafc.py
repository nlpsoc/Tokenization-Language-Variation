from unittest import TestCase
from styletokenizer.utility.gyafc import load_train_data, load_dev_data


class Test(TestCase):
    def test_load_train_data(self):
        train_data = load_train_data()
        self.assertEqual(len(train_data["formal"]), len(train_data["informal"]))
        self.assertEqual(len(train_data["formal"]), 104562)  # statistic taken from https://aclanthology.org/N18-1012/

    def test_load_dev_data(self):
        """
            for now: only use the informal -> formal.ref0 pairs, i.e.,
            out of 5 options the direction of informal originals and one formal rewrite by crowd-workers
        """
        dev_data = load_dev_data()
        self.assertEqual(len(dev_data["formal"]), len(dev_data["informal"]))
        self.assertEqual(5665, len(dev_data["formal"]))  # statistic taken from https://aclanthology.org/N18-1012/
