from unittest import TestCase

import datasets

from styletokenizer.utility import umich_av


class Test(TestCase):
    def test_load_data(self):
        dataset = umich_av.load_1_dev_data()
        print(dataset)
        self.assertTrue(type(dataset) == datasets.Dataset)

    def test_create_pairs(self):
        dataset = umich_av.load_1_dev_data()
        pairs, labels = umich_av._create_pairs(dataset)

        self.assertTrue(len(dataset) < len(labels))
        self.assertTrue(len(labels) == len(dataset)*2)
        self.assertTrue(len(pairs) == len(labels))
        # labels has as many 0s as 1s
        self.assertTrue(labels.count(0) == labels.count(1))
