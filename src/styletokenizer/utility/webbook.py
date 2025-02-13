"""
    get path to the webbook corpus
"""
from styletokenizer.utility.env_variables import at_uu, at_umich

if at_uu():
    FOLDER_BASE = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data"
elif at_umich():
    FOLDER_BASE = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data"
else:
    FOLDER_BASE = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data"
CORPORA_WEBBOOK = f"{FOLDER_BASE}/train-corpora/webbook"
