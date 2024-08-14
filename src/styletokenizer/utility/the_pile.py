import os
import zstandard as zstd
import json
from styletokenizer.utility.custom_logger import log_and_flush
import re

PILE_SET_NAMES = ['Gutenberg (PG-19)', 'StackExchange', 'OpenSubtitles', 'Github', 'Pile-CC', 'DM Mathematics']
WORD_COUNTS = [50000000,
               200000000,
               50000000,
               50000000,
               100000000,
               20000000]


def read_lines_from_zst(file_path):
    """Generator to read lines from a .zst compressed file"""
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buffer = ''
            while True:
                chunk = reader.read(1024)  # Read in chunks of 1024 bytes
                if not chunk:
                    break
                buffer += chunk.decode('utf-8')
                lines = buffer.split('\n')
                buffer = lines.pop()
                for line in lines:
                    yield line


def sample_pile_texts(pile_set_names=PILE_SET_NAMES, word_counts=WORD_COUNTS, test=False, individual_text_length=None):
    dir_path = "/shared/4/datasets/thepile/pile/train"
    zst_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl.zst')]
    log_and_flush(zst_files)

    sampled_items = []

    # Dictionary to keep track of the word count for each pile set name
    word_counts_dict = dict(zip(pile_set_names, word_counts))
    log_and_flush(word_counts_dict)
    current_word_counts = {name: 0 for name in pile_set_names}
    log_and_flush(current_word_counts)
    log_and_flush(pile_set_names)

    line_counter = 0

    def should_continue_sampling():
        """Check if we need to continue sampling, so ANY requirement not yet fulfilled"""
        return any(current_word_counts[name] < word_counts_dict[name] for name in pile_set_names)

    for filename in zst_files:
        if not should_continue_sampling():
            break

        file_path = os.path.join(dir_path, filename)
        log_and_flush(f"Sampling from {file_path}")
        for line in read_lines_from_zst(file_path):
            if not should_continue_sampling():
                break

            try:
                data = json.loads(line)
                pile_set_name = data.get('meta', {}).get('pile_set_name')
                if (pile_set_name in pile_set_names) and (
                        current_word_counts[pile_set_name] < word_counts_dict[pile_set_name]):
                    text = data.get('text', '')
                    if individual_text_length:  # we need samples of an exact length
                        tokens = re.findall(r'\S+|\s+', text)
                        text_word_count = int(len(tokens) / 2)
                        if text_word_count < individual_text_length:
                            continue
                        else:
                            text = ''.join(tokens[:individual_text_length * 2])
                            text_word_count = individual_text_length

                        sampled_items.append({"id": line_counter, "text": text, "word_count": text_word_count,
                                              "source": pile_set_name})
                    else:
                        text_word_count = len(text.split())

                        sampled_items.append({"id": line_counter, "text": text, "word_count": text_word_count,
                                              "domain": pile_set_name, "source": "thePile"})
                    current_word_counts[pile_set_name] += text_word_count
            except json.JSONDecodeError:
                log_and_flush("decode error")
                continue
            line_counter += 1

            if test:
                break
        log_and_flush(f"Sampled {current_word_counts} total")

    # Return the sampled data
    return sampled_items
