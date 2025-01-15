import csv
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import random


# Make sure you've downloaded NLTK’s punkt tokenizer once:
# nltk.download('punkt')

def nltk_split_into_sentences(paragraph):
    """
    Splits a paragraph into sentences using NLTK's sent_tokenize.
    Returns a list of (start_char, end_char, sentence_text) tuples,
    where start_char/end_char are offsets in the original paragraph.
    """
    sentences_raw = nltk.sent_tokenize(paragraph)

    sentences_with_offsets = []
    search_start_idx = 0
    for sent in sentences_raw:
        sent_str = sent.strip()
        idx = paragraph.find(sent_str, search_start_idx)
        if idx == -1:
            # fallback if something unexpected, but usually should not happen
            continue
        start_char = idx
        end_char = idx + len(sent_str)
        sentences_with_offsets.append((start_char, end_char, sent_str))
        # Move the search start beyond this sentence
        search_start_idx = end_char

    return sentences_with_offsets


def extract_mistakes_from_sgml(filename, csv_out):
    """
    Reads the SGML file, locates mistakes, identifies the containing sentence,
    and writes out a CSV with columns:
      - sentence
      - type
      - mistake
      - correction
      - character span
      - sentence_id

    Every sentence (with or without mistakes) will appear in the CSV:
      - If no mistakes are in that sentence, the fields for type/mistake/correction/character span
        will be empty.
      - If multiple mistakes occur in the same sentence, that sentence will appear multiple times,
        once for each mistake.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        sgml_data = f.read()

    # If you prefer not to install lxml, you can use "html.parser" here:
    soup = BeautifulSoup(sgml_data, 'lxml')

    # We'll collect rows for the CSV in a list of lists or a list of dicts.
    rows_for_csv = []

    # For each <DOC> ...
    for doc in soup.find_all('doc'):
        doc_id = doc.get("nid", "")
        text_tag = doc.find('text')
        if not text_tag:
            continue

        # Extract paragraphs (the <p> tags).
        paragraphs = text_tag.find_all('p')
        has_title = len(text_tag.find_all('title')) > 0

        # Convert them into a list of "clean" paragraph strings,
        # removing extra newlines/spaces.
        paragraph_texts = []
        for p in paragraphs:
            ptext = p.get_text()
            ptext = ptext.strip()
            paragraph_texts.append(ptext)

        # --- Collect all mistakes in this DOC ---
        # We'll store them in a dict keyed by paragraph index (0-based after we adjust).
        # Each value is a list of mistakes, where each mistake is a dict with:
        #   start_off, end_off, mistake_text, correction_text, error_type
        paragraph_mistakes = {}

        annotation_tag = doc.find('annotation')
        if annotation_tag:
            for mistake_tag in annotation_tag.find_all('mistake'):
                # SGML sometimes has may have paragraphs numbered from 1, if there is a title before, otherwise 0
                start_par = int(mistake_tag['start_par'])
                if has_title:
                    start_par -= 1

                start_off = int(mistake_tag['start_off'])
                end_off = int(mistake_tag['end_off'])

                error_type = mistake_tag.find('type').get_text(strip=True)
                correction = mistake_tag.find('correction').get_text(strip=True)

                # The "mistake" text is presumably in one paragraph
                paragraph_text = paragraph_texts[start_par]
                mistake_text = paragraph_text[start_off:end_off]

                if start_par not in paragraph_mistakes:
                    paragraph_mistakes[start_par] = []

                paragraph_mistakes[start_par].append({
                    'start_off': start_off,
                    'end_off': end_off,
                    'error_type': error_type,
                    'correction': correction,
                    'mistake_text': mistake_text
                })

        # --- Now process each paragraph, sentence by sentence ---
        for p_idx, paragraph_text in enumerate(paragraph_texts):
            sentences = nltk_split_into_sentences(paragraph_text)
            # If no sentences found (maybe paragraph is empty), just skip
            if not sentences:
                continue

            # All mistakes for this paragraph (if any)
            mistakes_in_par = paragraph_mistakes.get(p_idx, [])

            for (sent_start, sent_end, sent_text) in sentences:
                # We want to see which mistakes fall into [sent_start, sent_end)
                # start_off >= sent_start and end_off <= sent_end
                # Because multiple mistakes can occur in a single sentence, we gather them all.
                mistakes_this_sentence = [
                    m for m in mistakes_in_par
                    if m['start_off'] >= sent_start and m['end_off'] <= sent_end
                ]

                if not mistakes_this_sentence:
                    # If no mistakes in this sentence, output one row with empty fields for them
                    rows_for_csv.append([
                        sent_text,  # sentence
                        "",  # type
                        "",  # mistake
                        "",  # correction
                        "",  # character span
                        f"{doc_id}-{p_idx + 1}-{sent_start}"  # sentence_id
                    ])
                else:
                    # One or more mistakes in this sentence => output multiple rows
                    for m in mistakes_this_sentence:
                        offset_in_sentence = m['start_off'] - sent_start
                        mistake_length = m['end_off'] - m['start_off']
                        span_start = offset_in_sentence
                        span_end = offset_in_sentence + mistake_length
                        char_span_str = f"{span_start}:{span_end}"

                        rows_for_csv.append([
                            sent_text.strip(),  # sentence
                            m['error_type'],  # type
                            m['mistake_text'],  # mistake
                            m['correction'],  # correction
                            char_span_str,  # character span
                            f"{doc_id}-{p_idx + 1}-{sent_start}"  # sentence_id
                        ])

    # Create a Pandas DataFrame
    df = pd.DataFrame(rows_for_csv, columns=[
        'sentence', 'type', 'mistake', 'correction', 'character span', 'sentence_id'
    ])
    df.to_csv(csv_out, index=False, encoding='utf-8')

    balanced_sentences_df = collapse_and_balance_sentences(df)
    # print how many sentences are in each label
    print(balanced_sentences_df["label"].value_counts())

    return balanced_sentences_df
    # # split into train, dev, test as 80/10/10
    # train_size = int(0.8 * len(balanced_sentences_df))
    # dev_size = int(0.1 * len(balanced_sentences_df))
    # train_df = balanced_sentences_df[:train_size]
    # dev_df = balanced_sentences_df[train_size:train_size + dev_size]
    # test_df = balanced_sentences_df[train_size + dev_size:]
    # train_df.to_csv("train.csv", index=False, encoding="utf-8")
    # dev_df.to_csv("dev.csv", index=False, encoding="utf-8")
    # test_df.to_csv("test.csv", index=False, encoding="utf-8")


def collapse_and_balance_sentences(df):
    """
    Takes the initial DataFrame (with columns like 'sentence', 'type', 'sentence_id'),
    collapses all mistakes by sentence, then balances the classes (0 vs. 1),
    and finally shuffles the result.

    Returns a new DataFrame with columns:
      - 'sentence'
      - 'sentence_id'
      - 'types'  (comma-separated list of unique error types)
      - 'label'  ('0' if no mistakes, '1' if at least one)
    """

    # 1) Define helper functions

    def collapse_types(type_series):
        """
        Given a Series of error types (some might be empty strings),
        return a comma-separated list of unique non-empty error types.
        """
        unique_types = set(t.strip() for t in type_series if t.strip())
        return ",".join(sorted(unique_types))  # sorted for consistency

    def has_mistake(type_series):
        """
        Return '1' if there's at least one non-empty type in the series, otherwise '0'.
        """
        for val in type_series:
            if val.strip():
                return '1'
        return '0'

    # 2) Group by "sentence_id" + "sentence", then aggregate
    grouped = (
        df.groupby(["sentence_id", "sentence"], as_index=False)
        .agg({"type": [collapse_types, has_mistake]})
    )

    # Flatten multi-level columns
    grouped.columns = ["sentence_id", "sentence", "types", "label"]

    # Reorder columns if desired
    grouped = grouped[["sentence", "sentence_id", "types", "label"]]

    # 3) Balance classes
    df_0 = grouped[grouped["label"] == "0"]
    df_1 = grouped[grouped["label"] == "1"]
    count_0 = len(df_0)
    count_1 = len(df_1)

    if count_0 == 0 or count_1 == 0:
        # If there's no data in one of the classes, skip balancing
        balanced_df = grouped
    else:
        minority_size = min(count_0, count_1)
        # Sample from each class
        df_0_bal = df_0.sample(n=minority_size, random_state=42)
        df_1_bal = df_1.sample(n=minority_size, random_state=42)
        balanced_df = pd.concat([df_0_bal, df_1_bal], ignore_index=True)

    # 4) Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def build_pairs(df):
    # type_counts = df["types"].str.split(",").explode().value_counts()
    # print(type_counts) --> distribution of error types is very imbalanced; as a result will not try to make balanced pairs

    # change df to only contain a maxmimum of 5k sentences with no types
    df_no_types = df[df["types"] == ""]
    df_with_types = df[df["types"] != ""]
    df_no_types = df_no_types.sample(n=5000, random_state=42)
    df = pd.concat([df_no_types, df_with_types], ignore_index=True)

    # split into 80/10/10
    train_size = int(0.8 * len(df))
    dev_size = int(0.1 * len(df))
    train_df = df[:train_size]
    dev_df = df[train_size:train_size + dev_size]
    test_df = df[train_size + dev_size:]

    def check_overlap(types1, types2):
        return bool(set(types1.split(',')) & set(types2.split(',')))

    for i, overlap_size in enumerate([25_000 * 0.8, 25_000 * 0.1, 25_000 * 0.1]):
        overlap_pairs = []
        non_overlap_pairs = []

        # Iteratively sample pairs
        while len(overlap_pairs) < overlap_size or len(non_overlap_pairs) < overlap_size:
            # Randomly sample two rows
            row1, row2 = df.sample(n=2).itertuples(index=False)

            # Check overlap
            overlap = check_overlap(row1.types, row2.types)

            if overlap and len(overlap_pairs) < overlap_size:
                overlap_pairs.append({
                    'sentence1': row1.sentence,
                    'sentence2': row2.sentence,
                    'types1': row1.types,
                    'types2': row2.types,
                    'sentence1_id': row1.sentence_id,
                    'sentence2_id': row2.sentence_id,
                    'Error Overlap': 1
                })

            elif not overlap and len(non_overlap_pairs) < overlap_size:
                non_overlap_pairs.append({
                    'sentence1': row1.sentence,
                    'sentence2': row2.sentence,
                    'types1': row1.types,
                    'types2': row2.types,
                    'sentence1_id': row1.sentence_id,
                    'sentence2_id': row2.sentence_id,
                    'Error Overlap': 0
                })

        # build DataFrame out of the pairs
        pairs_df = pd.DataFrame(overlap_pairs + non_overlap_pairs)
        # shuffle
        pairs_df = pairs_df.sample(frac=1).reset_index(drop=True)
        split = "train" if i == 0 else "dev" if i == 1 else "test"
        # save to csv
        pairs_df.to_csv(f"overlap_pairs_{split}.csv", index=False, encoding="utf-8")

        # print occurence per IDs
        print(pd.concat([pairs_df["sentence1_id"], pairs_df["sentence2_id"]]).value_counts())
        # occurence of types
        print(pd.concat(
            [pairs_df["types1"].str.split(",").explode(), pairs_df["types1"].str.split(",").explode()]).value_counts())


if __name__ == "__main__":
    # Usage example
    sgml_file = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/NUCLE/release3.3/data/nucle3.2.sgml"  # Your SGML input file
    csv_output = "output.csv"  # Desired output CSV
    # set seed
    random.seed(42)
    balanced_sentences = extract_mistakes_from_sgml(sgml_file, csv_output)
    build_pairs(balanced_sentences)
    print(f"CSV written to {csv_output}")
