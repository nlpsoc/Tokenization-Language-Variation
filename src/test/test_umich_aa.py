from unittest import TestCase
import styletokenizer.utility.umich_aa as umich_aa


class Test(TestCase):
    def test_load_dev_data(self):
        data = umich_aa.load_dev_data()

        print(data["author"].unique())
        print(data["authorIDs"])

        # get the unique author counts
        print(data['authorIDs'].value_counts())


