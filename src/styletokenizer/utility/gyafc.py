import pandas as pd

GYAFC_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/GYAFC/GYAFC_Corpus/"
ENTERTAINMENT = "Entertainment_Music/"
FAMILY = "Family_Relationships/"


def load_train_data():
    entertainment_df = _get_theme_data(ENTERTAINMENT)
    family_df = _get_theme_data(FAMILY)
    return pd.concat([entertainment_df, family_df], axis=0)


def load_dev_data():
    entertainment_df = _get_theme_data(ENTERTAINMENT, split="dev")
    family_df = _get_theme_data(FAMILY, split="dev")
    return pd.concat([entertainment_df, family_df], axis=0)


def _get_theme_data(theme_dir, split="train"):
    # theme dir is either ENTERTAINMENT or FAMILY
    assert ((theme_dir == ENTERTAINMENT) or (theme_dir == FAMILY))
    # split is either train or dev
    assert (split == "train") or (split == "dev")
    formal = "formal"
    informal = "informal"
    if split == "dev":
        split = "tune"
        formal = "formal.ref0"
    formal_lines = _read_txt_lines(GYAFC_PATH + theme_dir + split + "/" + formal)
    informal_lines = _read_txt_lines(GYAFC_PATH + theme_dir + split + "/" + informal)
    # Create a DataFrame with two columns using the lists
    entertainment_df = pd.DataFrame({
        'formal': formal_lines,
        'informal': informal_lines
    })
    return entertainment_df


"""
    utility
"""


def _read_txt_lines(document_path):
    with open(document_path, 'r') as file:
        document_lines = file.readlines()
    return document_lines


def to_classification_data(train_data):
    # create classification data
    train_texts = train_data["formal"].tolist() + train_data["informal"].tolist()
    train_labels = [1 for _ in range(len(train_data))] + [0 for _ in range(len(train_data))]
    return train_labels, train_texts
