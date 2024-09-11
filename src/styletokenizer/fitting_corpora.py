from styletokenizer.utility.env_variables import at_uu

if at_uu():
    FOLDER_BASE = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER"
else:
    FOLDER_BASE = "/shared/3/projects/hiatus/TOKENIZER_wegmann"
CORPORA_MIXED = f"{FOLDER_BASE}/data/fitting-corpora/mixed"
CORPORA_TWITTER = f"{FOLDER_BASE}/data/fitting-corpora/twitter"
CORPORA_WIKIPEDIA = f"{FOLDER_BASE}/data/fitting-corpora/wikipedia"
