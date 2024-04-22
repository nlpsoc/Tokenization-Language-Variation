from unittest import TestCase
from styletokenizer.load_data import load_pickle_file
from styletokenizer.utility.filesystem import get_dir_to_src


class Test(TestCase):
    def test_load_pickle_file(self):
        """
            original small reddit test data has the form
                Test load_pickle_file function
        :return:
        """
        small_reddit_dev = (get_dir_to_src() + "/../data/development/dev_reddit-corpus-small-30000_dataset.pickle")
        data = load_pickle_file(small_reddit_dev)
        self.assertTrue(type(data) == list)
        self.assertTrue(type(data[0]) == dict)
        # check if the first entry has the right keys: label, u1, u1_id, u2, u2_id, u1_speaker, u2_speaker, same_conv
        self.assertTrue("label" in data[0].keys())
        self.assertTrue("u1" in data[0].keys())
        self.assertTrue("u1_id" in data[0].keys())
        self.assertTrue("u2" in data[0].keys())
        self.assertTrue("u2_id" in data[0].keys())
        self.assertTrue("u1_speaker" in data[0].keys())
        self.assertTrue("u2_speaker" in data[0].keys())
        self.assertTrue("same_conv" in data[0].keys())

