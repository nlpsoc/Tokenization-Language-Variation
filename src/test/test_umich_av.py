import re
from unittest import TestCase

import datasets

from styletokenizer.utility import umich_av
from whitespace_consts import APOSTROPHES, APOSTROPHE_PATTERN


class Test(TestCase):
    def test_load_data(self):
        dataset = umich_av.load_1_dev_data()
        len(dataset)
        print(dataset)
        self.assertTrue(type(dataset) == datasets.Dataset)

    def test_create_pairs(self):
        dataset = umich_av.load_1_dev_data()
        pairs, labels = umich_av._create_pairs(dataset)

        self.assertTrue(len(dataset) < len(labels))
        self.assertTrue(len(labels) == len(dataset) * 2)
        self.assertTrue(len(pairs) == len(labels))
        # labels has as many 0s as 1s
        self.assertTrue(labels.count(0) == labels.count(1))

    def test_ws_variation(self):
        dataset = umich_av.get_1_train_dataframe()
        apostrophe_pattern = APOSTROPHE_PATTERN

        # Function to find and extract context around apostrophes in a column
        def extract_apostrophe_context_with_unicode(text, pattern, context=5):
            matches = []
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - context)
                end = min(len(text), match.end() + context)
                context_str = text[start:match.start()] + match.group() + " (U+" + format(ord(match.group()),
                                                                                          '04X') + ")" + text[
                                                                                                         match.end():end]
                matches.append((match.group(), context_str))
            return matches

        def find_apostrophes(df, column_name, pattern):
            df['apostrophe_context'] = df[column_name].apply(lambda x: extract_apostrophe_context_with_unicode(x, pattern))
            return df

        result_df = find_apostrophes(dataset, 'query', apostrophe_pattern)
        result_df = result_df[result_df['apostrophe_context'].apply(bool)]
        # Explode the context column to separate rows for each match
        exploded_df = result_df.explode('apostrophe_context')

        # Extract the apostrophe and context separately
        exploded_df['apostrophe'] = exploded_df['apostrophe_context'].apply(lambda x: x[0])
        exploded_df['context'] = exploded_df['apostrophe_context'].apply(lambda x: x[1])

        # Group by apostrophe type and collect examples
        grouped = exploded_df.groupby('apostrophe')['context'].apply(list).reset_index()

        # Function to print number of examples and up to 10 examples per apostrophe type
        def print_examples_per_apostrophe_type(grouped_df, max_examples=10):
            for index, row in grouped_df.iterrows():
                apostrophe = row['apostrophe']
                examples = row['context']
                num_examples = len(examples)
                print(f"Apostrophe: {apostrophe} (U+{ord(apostrophe):04X}) - {num_examples} examples")
                for example in examples[:max_examples]:
                    print(f"  Example: {example}")
                print()

        # Display the examples
        print_examples_per_apostrophe_type(grouped)