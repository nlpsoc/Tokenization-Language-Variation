"""
    script to see how many ws variations are present in the UMICH AV dataset
"""

from styletokenizer.utility import umich_av
from styletokenizer.whitespace_consts import WHITESPACE_PATTERN
from utility.umich_av import find_av_matches

if __name__ == "__main__":
    df = umich_av.get_10_train_dataframe()
    ws_pattern = WHITESPACE_PATTERN

    find_av_matches(df, ws_pattern)
