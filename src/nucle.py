import re
import csv
from bs4 import BeautifulSoup


def split_into_sentences(paragraph):
    """
    Splits a paragraph into sentences *naively* by ., ?, or !.
    Returns a list of (start_char, end_char, sentence_text) tuples,
    where start_char/end_char are offsets in the original paragraph.
    """
    # We want to keep track of offsets, so we'll do it manually.
    # A simple approach:
    sentences = []
    current_start = 0

    # Find all positions where there's a sentence-ending punctuation.
    # We'll also keep the punctuation with the sentence for clarity.
    for match in re.finditer(r'[.?!]', paragraph):
        # `match.start()` is the position of the punctuation
        end_pos = match.start() + 1  # +1 to include the punctuation
        sentence_text = paragraph[current_start:end_pos].strip()
        if sentence_text:
            sentences.append((current_start, end_pos, sentence_text))
        current_start = end_pos

    # If there's leftover text after the last punctuation, add it as well.
    if current_start < len(paragraph):
        leftover = paragraph[current_start:].strip()
        if leftover:
            sentences.append((current_start, len(paragraph), leftover))

    return sentences


def extract_mistakes_from_sgml(filename, csv_out):
    """
    Reads the SGML file, locates mistakes, identifies the containing sentence,
    and writes out the CSV with columns: sentence, mistake, correction, character span.
    """
    # 1. Read the SGML file
    with open(filename, 'r', encoding='utf-8') as f:
        sgml_data = f.read()

    # 2. Parse using BeautifulSoup (lxml or html.parser)
    soup = BeautifulSoup(sgml_data, 'lxml')

    # Prepare a list of rows for our CSV
    rows_for_csv = []

    # 3. For each <DOC> ...
    for doc in soup.find_all('doc'):
        text_tag = doc.find('text')
        if not text_tag:
            continue

        # 4. Extract paragraphs (the <p> tags) as a list of strings
        paragraphs = text_tag.find_all('p')
        paragraph_texts = [p.get_text() for p in paragraphs]

        annotation_tag = doc.find('annotation')
        if not annotation_tag:
            # Some <DOC> might have no <ANNOTATION>
            continue

        # 5. For each <MISTAKE> ...
        for mistake_tag in annotation_tag.find_all('mistake'):
            # Read start/end paragraph, start/end offset
            start_par = int(mistake_tag['start_par'])
            end_par = int(mistake_tag['end_par'])
            start_off = int(mistake_tag['start_off'])
            end_off = int(mistake_tag['end_off'])

            # Read the correction text
            correction = mistake_tag.find('correction').get_text(strip=True)

            # 6. Extract the relevant substring (the 'mistake') from the text
            #    (assuming the mistake is contained in a single paragraph)
            paragraph_text = paragraph_texts[start_par - 1].strip()
            mistake_text = paragraph_text[start_off:end_off]

            # 7. Identify which sentence this substring belongs to
            sentences = split_into_sentences(paragraph_text)
            sentence_found = None
            char_span_in_sentence = (None, None)

            for (sent_start, sent_end, sent_text) in sentences:
                if (start_off >= sent_start) and (end_off <= sent_end):
                    # Found the sentence containing the error
                    sentence_found = sent_text
                    # Now find offsets *within* that sentence
                    offset_in_sentence = start_off - sent_start
                    char_span_in_sentence = (
                        offset_in_sentence,
                        offset_in_sentence + (end_off - start_off)
                    )
                    break

            # If for some reason we couldn't find a sentence,
            # just use the entire paragraph (fallback)
            if not sentence_found:
                sentence_found = paragraph_text
                # character span = from start_off to end_off in the paragraph
                char_span_in_sentence = (start_off, end_off)

            # 8. Prepare a row for the CSV
            # character_span_str will look like "start:end"
            char_span_str = f"{char_span_in_sentence[0]}:{char_span_in_sentence[1]}"

            # assert that the char span still equals the mistake text
            assert sentence_found[char_span_in_sentence[0]:char_span_in_sentence[1]] == mistake_text

            rows_for_csv.append([
                sentence_found.strip(),
                mistake_text,
                correction,
                char_span_str
            ])

    # 9. Write out the CSV
    with open(csv_out, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        # Write header
        writer.writerow(["sentence", "mistake", "correction", "character span"])
        # Write rows
        writer.writerows(rows_for_csv)


if __name__ == "__main__":
    # Usage example
    sgml_file = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/NUCLE/release3.3/data/nucle3.2.sgml"  # Your SGML input file
    csv_output = "output.csv"  # Desired output CSV
    extract_mistakes_from_sgml(sgml_file, csv_output)
    print(f"CSV written to {csv_output}")