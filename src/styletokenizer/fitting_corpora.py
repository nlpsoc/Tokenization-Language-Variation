from styletokenizer.utility.env_variables import at_uu, at_umich

if at_uu():
    FOLDER_BASE = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER"
elif at_umich():
    FOLDER_BASE = "/shared/3/projects/hiatus/TOKENIZER_wegmann"
else:
    FOLDER_BASE = "/Users/anna/Documents/git projects.nosync/StyleTokenizer"
CORPORA_MIXED = f"{FOLDER_BASE}/data/fitting-corpora/mixed"
CORPORA_TWITTER = f"{FOLDER_BASE}/data/fitting-corpora/twitter"
CORPORA_WIKIPEDIA = f"{FOLDER_BASE}/data/fitting-corpora/wikipedia"
CORPORA_PUBMED = f"{FOLDER_BASE}/data/fitting-corpora/pubmed"
