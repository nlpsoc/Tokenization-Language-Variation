from unittest import TestCase
import styletokenizer.utility.umich_aa as umich_aa


class Test(TestCase):
    def test_load_dev_data(self):
        data = umich_aa.load_dev_data()

        # print(data["author"].unique())
        # print(data["authorIDs"])

        # get the unique author counts
        print(data['authorIDs'].value_counts())

    def test_split_data(self):
        data = umich_aa.load_dev_data()
        train, dev, test = umich_aa.split_df(data)

        # assert that all authorIDs in train also occur in dev and test
        train_authors = set(train['authorIDs'].unique())
        dev_authors = set(dev['authorIDs'].unique())
        test_authors = set(test['authorIDs'].unique())
        print(f"Train: {len(train_authors)} Dev: {len(dev_authors)} Test: {len(test_authors)}")
        self.assertTrue(train_authors.issubset(dev_authors))
        self.assertTrue(train_authors.issubset(test_authors))
        self.assertTrue(dev_authors.issubset(test_authors))

        # assert that each author is similarly represented in train, dev and test
        train_author_counts = train['authorIDs'].value_counts()
        dev_author_counts = dev['authorIDs'].value_counts()
        test_author_counts = test['authorIDs'].value_counts()
        self.assertTrue((train_author_counts - dev_author_counts*(train_author_counts[0]/dev_author_counts[0])).abs().max() < 5)
        self.assertTrue((train_author_counts - test_author_counts*(train_author_counts[0]/test_author_counts[0])).abs().max() < 5)
        self.assertTrue((dev_author_counts - test_author_counts).abs().max() < 2)

        # get the average number of posts per author
        print(f"Train: {train_author_counts.mean()} Dev: {dev_author_counts.mean()} Test: {test_author_counts.mean()}")



