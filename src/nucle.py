import csv
import nltk
from bs4 import BeautifulSoup
import pandas as pd

# Make sure you've downloaded NLTK’s punkt tokenizer once:
# nltk.download('punkt')

def nltk_split_into_sentences(paragraph):
    """
    Splits a paragraph into sentences using NLTK's sent_tokenize.
    Returns a list of (start_char, end_char, sentence_text) tuples,
    where start_char/end_char are offsets in the original paragraph.
    """
    # sent_tokenize returns a list of sentence strings, but not offsets.
    # We'll re-find each sentence in the paragraph to get offsets.

    # 1) Use sent_tokenize to get raw sentence strings
    sentences_raw = nltk.sent_tokenize(paragraph)

    # 2) For each sentence, find its start index in the paragraph
    #    starting from the last matched index.
    sentences_with_offsets = []
    search_start_idx = 0
    for sent in sentences_raw:
        # Trim leading/trailing whitespace from the sentence
        sent_str = sent.strip()
        # Find the index of this sentence in paragraph (starting search from `search_start_idx`).
        # This is somewhat naive if repeated text occurs, but typically works well.
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
    and writes out a CSV with columns: sentence, mistake, correction, character span.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        sgml_data = f.read()

    soup = BeautifulSoup(sgml_data, 'lxml')  # or "html.parser" if you prefer

    rows_for_csv = []

    # For each <DOC> ...
    for doc in soup.find_all('doc'):
        text_tag = doc.find('text')
        if not text_tag:
            continue

        # Extract paragraphs (the <p> tags). We re-index them from 1 for convenience
        paragraphs = text_tag.find_all('p')

        # Convert them into a list of "clean" paragraph strings,
        # removing extra newlines/spaces.
        paragraph_texts = []
        for p in paragraphs:
            ptext = p.get_text()
            # Normalize whitespace: remove newlines, collapse multiple spaces, strip ends
            ptext = ptext.strip()
            paragraph_texts.append(ptext)

        annotation_tag = doc.find('annotation')
        if not annotation_tag:
            # Some <DOC> might have no <ANNOTATION>
            continue

        # For each <MISTAKE> ...
        for mistake_tag in annotation_tag.find_all('mistake'):
            # Paragraph indices in the SGML start from 0 or 1?
            # Let’s assume they start from 0 in the original file, but you said you want them from 1.
            # If your data *already* says "start_par=1", "end_par=1" for the first paragraph,
            # no shift is needed. If your data says "start_par=0" for the first paragraph,
            # then you’d need +1.
            start_par = int(mistake_tag['start_par']) - 1
            end_par = int(mistake_tag['end_par'])

            # If your data actually intends paragraphs to start from 1,
            # but your code is indexing from 0, subtract 1 to align:
            # start_par -= 1
            # end_par -= 1

            start_off = int(mistake_tag['start_off'])
            end_off = int(mistake_tag['end_off'])

            # Correction text
            correction = mistake_tag.find('correction').get_text(strip=True)

            # Error Type
            error_type = mistake_tag.find('type').get_text(strip=True)

            # The "mistake" text is presumably in one paragraph
            paragraph_text = paragraph_texts[start_par]
            mistake_text = paragraph_text[start_off:end_off]

            # Use NLTK to split the paragraph into sentences with offsets
            sentences = nltk_split_into_sentences(paragraph_text)

            sentence_found = None
            char_span_in_sentence = (None, None)

            sentence_offset = 0
            for (sent_start, sent_end, sent_text) in sentences:
                # Check if the mistake is fully within this sentence’s range
                if start_off >= sent_start and end_off <= sent_end:
                    # Found the containing sentence
                    sentence_found = sent_text
                    sentence_offset = sent_start
                    offset_in_sentence = start_off - sent_start
                    char_span_in_sentence = (
                        offset_in_sentence,
                        offset_in_sentence + (end_off - start_off)
                    )
                    break

            # If we never found the sentence, fall back to using the whole paragraph
            if not sentence_found:
                sentence_found = paragraph_text
                char_span_in_sentence = (start_off, end_off)

            # For debugging, you can check if the extracted substring matches:
            # (It may fail if there's whitespace mismatch or punctuation differences.)
            substring_in_sentence = sentence_found[
                char_span_in_sentence[0] : char_span_in_sentence[1]
            ]
            assert substring_in_sentence == mistake_text, (
                f"Mismatch: found '{substring_in_sentence}', expected '{mistake_text}'"
            )

            # Prepare row for CSV
            # character_span_str is "start:end" within the sentence
            start_char_sen, end_char_sen = char_span_in_sentence
            char_span_str = f"{start_char_sen}:{end_char_sen}"
            rows_for_csv.append([
                sentence_found.strip(),
                error_type,
                mistake_text,
                correction,
                char_span_str,
                f"{start_par+1}-{sentence_offset}"  # paragraph ID
            ])

    df = pd.DataFrame(rows_for_csv, columns=['sentence', 'type', 'mistake', 'correction', 'character span', "sentence id"])
    df.to_csv(csv_out, index=False, encoding='utf-8')

    # # Write the CSV
    # with open(csv_out, 'w', newline='', encoding='utf-8') as out_f:
    #     writer = csv.writer(out_f)
    #     writer.writerow(["sentence", "mistake", "correction", "character span"])
    #     writer.writerows(rows_for_csv)


if __name__ == "__main__":
    # Usage example
    sgml_file = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/NUCLE/release3.3/data/nucle3.2.sgml"  # Your SGML input file
    csv_output = "output.csv"  # Desired output CSV
    extract_mistakes_from_sgml(sgml_file, csv_output)
    print(f"CSV written to {csv_output}")