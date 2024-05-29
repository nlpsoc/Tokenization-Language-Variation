"""
    Script to see in how far different apostrophe variations are present in the UMICH AV dataset
"""

from styletokenizer.utility import umich_av
from styletokenizer.whitespace_consts import APOSTROPHE_PATTERN
from utility.umich_av import find_av_matches

if __name__ == "__main__":
    df = umich_av.get_1_train_dataframe()
    apostrophe_pattern = APOSTROPHE_PATTERN

    find_av_matches(df, apostrophe_pattern)
