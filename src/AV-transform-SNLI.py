from styletokenizer.utility.env_variables import set_cache

set_cache()
from datasets import load_dataset

import re
import string
# Define the punctuation set we care about
PUNCT = {'.', '!', '?'}
common_contractions = {
    "do not": "don't",
    "is not": "isn't",
    "are not": "aren't",
    "it is": "it's",
    "that is": "that's",
    "we are": "we're",
    "you are": "you're",
    "I am": "I'm",
    "I will": "I'll",
    "I would": "I'd",
    "they are": "they're",
    "will not": "won't",
    "can not": "can't",
    "there is": "there's"
}

def encased_with_apostrophes(text):
    # Check if the text is encased with standard quotes (artificat in SNLI)
    return text.startswith('"') and text.endswith('"')

def starts_with_uppercase_word(text):
    # Strip leading whitespace and check if the first character is uppercase
    text = text.lstrip()
    if not text:
        return False
    return text[0].isupper()

def ends_with_punctuation(text):
    # Check if the last non-whitespace character is punctuation
    text = text.rstrip()
    return len(text) > 0 and text[-1] in PUNCT

def contains_punctuation(text):
    # Check if there's any punctuation in the text
    # return any(ch in string.punctuation for ch in text)
    return any(ch in PUNCT for ch in text)

def whitespace_encoding(text):
    # Identify all distinct whitespace code points used in the text.
    # This will differentiate between e.g. U+0020 (normal space) and U+00A0 (no-break space).
    whitespaces = set()
    for ch in text:
        if ch.isspace():
            whitespaces.add(ord(ch))  # store the code point
    return whitespaces

def apostrophe_encoding(text):
    # Extract all apostrophe-like characters: common are `'` and `’`
    # Return a set of apostrophe chars used
    # If you want to be more comprehensive, include other variants.
    # Here we include backtick and right single quotation mark as well.
    possible_apostrophes = {"'", "’", "`"}
    apostrophes = {ch for ch in text if ch in possible_apostrophes}
    return apostrophes

def extract_number_patterns(text):
    # Find all numbers and their surrounding formatting.
    # We'll capture substrings around each digit sequence that may include punctuation and spacing.
    number_patterns = []
    for match in re.finditer(r"\d+", text):
        start, end = match.span()
        # Extend outwards to include punctuation/whitespace directly adjacent to the digits
        left = start
        while left > 0 and (text[left-1] in string.punctuation or text[left-1].isspace()):
            left -= 1
        right = end
        while right < len(text) and (text[right] in string.punctuation or text[right].isspace()):
            right += 1
        substring = text[left:right].strip()
        number_patterns.append(substring)
    return number_patterns

def compare_number_formats(patterns1, patterns2):
    # Check if both lists have the same number of numeric patterns
    if len(patterns1) != len(patterns2):
        return False
    # Compare each pair of patterns
    for p1, p2 in zip(patterns1, patterns2):
        # Compare digits sequence
        digits1 = re.sub(r"\D", "", p1)
        digits2 = re.sub(r"\D", "", p2)
        if digits1 != digits2:
            return False
        # Compare non-digit formatting
        non_digits1 = re.sub(r"\d", "", p1)
        non_digits2 = re.sub(r"\d", "", p2)
        if non_digits1 != non_digits2:
            return False
    return True

def contains_newline(text):
    return "\n" in text

def contains_contractions(text):
    # Check if text contains any of the known contracted forms
    pattern = r'\b(?:' + '|'.join(map(re.escape, common_contractions.values())) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def can_form_contractions(text):
    # Check if text contains any expansions that could be turned into known contractions
    # If we find at least one expansion pattern in the text, return True
    for expansion in common_contractions.keys():
        # Create a regex pattern for the expansion
        exp_words = expansion.split()
        pattern = r'\b' + r'\s+'.join(exp_words) + r'\b'
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def compare_texts(text1, text2):
    conditions = []
    conditions.append(encased_with_apostrophes(text1) == encased_with_apostrophes(text2))
    conditions.append(starts_with_uppercase_word(text1) == starts_with_uppercase_word(text2))
    conditions.append(ends_with_punctuation(text1) == ends_with_punctuation(text2))
    conditions.append(contains_punctuation(text1) == contains_punctuation(text2))
    conditions.append(whitespace_encoding(text1) == whitespace_encoding(text2))
    conditions.append(apostrophe_encoding(text1) == apostrophe_encoding(text2))
    patterns1 = extract_number_patterns(text1)
    patterns2 = extract_number_patterns(text2)
    conditions.append(compare_number_formats(patterns1, patterns2))
    conditions.append(contains_contractions(text1) == contains_contractions(text2))
    similarity = sum(conditions) / len(conditions)
    return similarity


def make_texts_similar(text1, text2, dialect_prob=0.5):
    # Now text1 and text2 should be similar in capitalization and end punctuation.
    # Apostrophe and whitespace encoding is the same initially.
    # Randomly decide if we want to change them for BOTH texts simultaneously.

    # Random chance to change the dialect for both texts
    if random.random() < dialect_prob:
        # Randomly select a dialect from DIALECTS
        attempts = 5  # limit attempts to avoid infinite loops
        changed = False
        org_text1 = " ".join(text1.split())
        org_text2 = " ".join(text2.split())
        while attempts > 0 and not changed:
            try:
                dialect = random.choice(DIALECTS)
                text1 = dialect.transform(text1 + ".\n" + text2)  # function seems to do different transformations depending on call
                text1, text2 = text1.split(".\n")
                text1 = " ".join(text1.split())  # function seems to sometimes add whitespaces
                text2 = " ".join(text2.split())
                if text1 != org_text1 and text2 != org_text2:
                    changed = True
            except:  # if the dialect transformation fails
                print(f"Failed to transform {text1} or {text2}. Retrying {attempts} more times ...")
            attempts -= 1

    # Adjust Quotes
    if encased_with_apostrophes(text1) != encased_with_apostrophes(text2):
        if encased_with_apostrophes(text1) and not encased_with_apostrophes(text2):
            text2 = '"' + text2 + '"'
        elif not encased_with_apostrophes(text1) and encased_with_apostrophes(text2):
            text2 = text2[1:-1]

    # Adjust capitalization at the start
    if starts_with_uppercase_word(text1) != starts_with_uppercase_word(text2):
        if starts_with_uppercase_word(text1) and not starts_with_uppercase_word(text2):
            stripped = text2.lstrip()
            if stripped:
                start_idx = len(text2) - len(stripped)
                text2 = text2[:start_idx] + stripped[0].upper() + stripped[1:]
        elif not starts_with_uppercase_word(text1) and starts_with_uppercase_word(text2):
            stripped = text2.lstrip()
            if stripped:
                start_idx = len(text2) - len(stripped)
                text2 = text2[:start_idx] + stripped[0].lower() + stripped[1:]

    # Adjust punctuation at the end
    if ends_with_punctuation(text1) != ends_with_punctuation(text2):
        if ends_with_punctuation(text1) and not ends_with_punctuation(text2):
            t1_end_punct = text1.rstrip()[-1]
            text2 = text2.rstrip() + t1_end_punct
        elif not ends_with_punctuation(text1) and ends_with_punctuation(text2):
            text2 = text2.rstrip()
            while text2 and text2[-1] in PUNCT:
                text2 = text2[:-1]

    # Random chance to change whitespace encoding for both
    # For example, replace all regular spaces with non-breaking spaces in both texts
    if random.random() < 0.5:
        # Check if we have spaces
        if " " in text1 or " " in text2:
            # Replace all spaces with non-breaking spaces
            text1 = text1.replace(" ", "\u00A0")
            text2 = text2.replace(" ", "\u00A0")

    # Random chance to toggle apostrophe encoding for both
    # If we have apostrophes, switch them from `'` to `’` or vice versa
    apos1 = apostrophe_encoding(text1)
    apos2 = apostrophe_encoding(text2)
    # Since they are initially the same, we can just pick a toggle.
    if random.random() < 0.5 and (apos1 and apos2):
        # If we have at least one type of apostrophe in the texts
        # If we find `'` in texts, replace it with `’`, else if `’` then replace with `'`
        if "'" in text1 or "'" in text2:
            # Replace `'` with `’`
            text1 = text1.replace("'", "’")
            text2 = text2.replace("'", "’")
        elif "’" in text1 or "’" in text2:
            # Replace `’` with `'`
            text1 = text1.replace("’", "'")
            text2 = text2.replace("’", "'")

    return text1, text2

from multivalue import Dialects
DIALECTS = [Dialects.ColloquialSingaporeDialect(), Dialects.AfricanAmericanVernacular(), Dialects.ChicanoDialect(), Dialects.IndianDialect(), Dialects.AppalachianDialect(),
            Dialects.NorthEnglandDialect(), Dialects.MalaysianDialect(), Dialects.AustralianDialect(), Dialects.HongKongDialect(), Dialects.NewZealandDialect(),
            Dialects.NigerianDialect(), Dialects.PakistaniDialect(), Dialects.PhilippineDialect(), Dialects.SoutheastAmericanEnclaveDialect()]

import random


def flip_quotes(t):
    if encased_with_apostrophes(t):
        return t[1:-1], True
    else:
        return '"' + t + '"', True


def flip_capitalization(t):
    stripped = t.lstrip()
    if not stripped:
        return t, False
    start_idx = len(t) - len(stripped)
    first_char = stripped[0]
    if first_char.isalpha():
        flipped = first_char.lower() if first_char.isupper() else first_char.upper()
        new_t = t[:start_idx] + flipped + stripped[1:]
        changed = (new_t != t)
        return new_t, changed
    else:
        return t, False


def toggle_end_punctuation(t):
    if ends_with_punctuation(t):
        original = t
        t = t.rstrip()
        while t and t[-1] in PUNCT:
            t = t[:-1]
        changed = (t != original)
        return t, changed
    else:
        return t + ".", True


# def toggle_punctuation_presence(t):
#     if contains_punctuation(t):
#         original = t
#         t = "".join(ch for ch in t if ch not in PUNCT).rstrip()
#         changed = (t != original)
#         return t, changed
#     else:
#         return t, False

def toggle_whitespace_encoding(t):
    # Assume it only includes " " whitespaces. Change those to non-breaking spaces (\u00A0)
    original = t
    if " " in t:
        # Replace all spaces with non-breaking spaces
        t = t.replace(" ", "\u00A0")
        changed = (t != original)
        return t, changed
    else:
        # No spaces to change
        return t, False


def toggle_apostrophe_encoding(t):
    original = t
    apos = apostrophe_encoding(t)
    if apos:
        if "'" in apos and "’" in apos:
            t = t.replace("'", "\uFFFF")
            t = t.replace("’", "'")
            t = t.replace("\uFFFF", "’")
        elif "'" in apos:
            t = t.replace("'", "’")
        elif "’" in apos:
            t = t.replace("’", "'")
        changed = (t != original)
        return t, changed
    else:
        return t, False


def toggle_number_format(t):
    patterns = extract_number_patterns(t)
    changed = False
    if patterns:
        for p in patterns:
            if ',' in p:
                new_p = re.sub(r",", "", p)
                if new_p != p:
                    idx = t.find(p)
                    if idx != -1:
                        t = t[:idx] + new_p + t[idx + len(p):]
                        changed = True
                        break
    return t, changed


def dialect_transform(text2):
    # randomly select a dialect from DIALECTS
    changed = False
    attempts = 5  # limit attempts to avoid infinite loops
    # transform
    while not changed and attempts > 0:
        dialect = random.choice(DIALECTS)
        transformed_text = text2
        try:
            transformed_text = dialect.transform(text2).strip()
        except:  # if the dialect transformation fails
            print(f"Failed to transform with {text2}. Retrying {attempts} more times...")
        if transformed_text != text2:
            return transformed_text, True
        attempts -= 1
    return text2, False


def maybe_add_contraction(text1, text2):
    # Only add a contraction if:
    # - text1 can form contractions
    # - text1 has no contractions
    # - text2 has no contractions
    original = text2
    if not can_form_contractions(text1):
        return text2, False
    if contains_contractions(text1) or contains_contractions(text2):
        return text2, False

    expansions = list(common_contractions.keys())
    random.shuffle(expansions)

    for expansion in expansions:
        exp_words = expansion.split()
        pattern = r'\b' + r'\s+'.join(exp_words) + r'\b'
        match = re.search(pattern, text2, flags=re.IGNORECASE)
        if match:
            contraction = common_contractions[expansion]
            matched_text = match.group(0)
            if matched_text[0].isupper():
                contraction = contraction[0].upper() + contraction[1:]
            text2 = text2[:match.start()] + contraction + text2[match.end():]
            return text2, (text2 != original)

    return text2, False


def make_texts_distinct(text1, text2, dialect_prob=0.5):
    """
        Assumes to be called on SNLI text pairs
    :param text1:
    :param text2:
    :return:
    """
    transformations = [
        flip_quotes,
        flip_capitalization,
        toggle_end_punctuation,
        toggle_whitespace_encoding,
        toggle_apostrophe_encoding,
        toggle_number_format,
        lambda t: maybe_add_contraction(text1, t),
    ]
    # flip coin which text to transform
    to_transform = random.choice([0, 1])

    # flip coin to to dialect_transform as this is a transformation that needs to run before all other transformations
    text_modified = [text1, text2][to_transform]
    if random.random() < dialect_prob:
        text_modified, changed = dialect_transform([text1, text2][to_transform])

    attempts = 20  # limit attempts to avoid infinite loops
    while attempts > 0:
        # Attempt two further random transformation
        three_trans = random.sample(transformations, 3)
        for transform in three_trans:
            new_text, changed = transform(text_modified)
            if changed:
                text_modified = new_text
        attempts -= 1
        if text_modified != [text1, text2][to_transform]:
            break
    result = [text1, text2]
    result[to_transform] = text_modified
    return result


import pandas as pd
import os
from tqdm import tqdm

dataset = load_dataset("snli")
os.makedirs("snli_modified", exist_ok=True)

for split in tqdm(dataset.keys(), desc="Processing splits"):
    print(f"Processing {split}")
    data = dataset[split]

    rows = []
    for example in tqdm(data, desc=f"Processing examples in {split}"):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = example["label"]

        # make sure that text is not empty
        if not premise or not hypothesis:
            continue
        # make sure that text is not N/A
        if pd.isna(premise) or pd.isna(hypothesis):
            continue

        # Skip if label is not in {0, 1, 2}
        if label not in {0, 1, 2}:
            continue

        # Flip a coin for similar/distinct
        want_similar = random.choice([True, False])

        if want_similar:
            # Make them similar
            premise, hypothesis = make_texts_similar(premise, hypothesis)
        else:
            # Make them distinct
            premise, hypothesis = make_texts_distinct(premise, hypothesis)

        style = 1 if want_similar else 0  # 1 for similar, 0 for distinct

        rows.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "premise_original": example["premise"],
            "hypothesis_original": example["hypothesis"],
            "nli": label,  # 0 entailment, 1 neutral, 2 contradiction
            "style": style  # 0 distinct, 1 similar
        })

    df = pd.DataFrame(rows,
                      columns=["premise", "hypothesis", "premise_original", "hypothesis_original", "nli", "style"])
    output_file = f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/{split}_modified.tsv"
    df.to_csv(output_file, index=False, encoding='utf-8', sep="\t")
