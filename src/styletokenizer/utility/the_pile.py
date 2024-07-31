import os
import zstandard as zstd
import json

PILE_SET_NAMES = ['Gutenberg (PG-19)', 'StackExchange', 'OpenSubtitles', 'Github', 'Pile-CC' , 'DM Mathematics']
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


def sample_pile_texts(pile_set_names=PILE_SET_NAMES, sampled_word_counts=WORD_COUNTS):
    dir_path = "/shared/4/datasets/thepile/pile/train"
    zst_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl.zst')]

    sampled_lines = []
    sampled_texts = []
    domains = []
    sampled_word_counts = []

    # Dictionary to keep track of the word count for each pile set name
    word_counts_dict = dict(zip(pile_set_names, sampled_word_counts))
    current_word_counts = {name: 0 for name in pile_set_names}

    line_counter = 0

    def should_continue_sampling():
        """Check if we need to continue sampling, so ANY requirement not yet fulfilled"""
        return any(current_word_counts[name] < word_counts_dict[name] for name in pile_set_names)

    for filename in zst_files:
        if not should_continue_sampling():
            break

        file_path = os.path.join(dir_path, filename)
        print(f"Sampling from {file_path}")
        for line in read_lines_from_zst(file_path):
            if not should_continue_sampling():
                break

            try:
                data = json.loads(line)
                pile_set_name = data.get('meta', {}).get('pile_set_name')
                if (pile_set_name in pile_set_names) and (
                        current_word_counts[pile_set_name] < word_counts_dict[pile_set_name]):
                    text = data.get('text', '')
                    text_word_count = len(text.split())
                    sampled_lines.append(line_counter)
                    sampled_texts.append(text)
                    domains.append(pile_set_name)
                    sampled_word_counts.append(text_word_count)
                    current_word_counts[pile_set_name] += text_word_count
            except json.JSONDecodeError:
                print("decode error")
                continue
            line_counter += 1
        print(f"Sampled {current_word_counts} total")

    # Return the sampled data
    return {
        "id": sampled_lines,
        "domain": domains,
        "source": ["thePile"] * len(sampled_texts),
        "word_count": sampled_word_counts,
        "text": sampled_texts,
    }
