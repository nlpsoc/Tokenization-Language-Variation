from styletokenizer.fitting_corpora import CORPORA_PUBMED
import styletokenizer.utility.the_pile as the_pile
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.datasets_helper import save_to_huggingface_format


def main(save_path=CORPORA_PUBMED, test=False):
    log_and_flush("Creating pubmed dataset")
    log_and_flush("Sampling from the pile")
    if test:
        pile_sample_dicts = the_pile.sample_pile_texts(ensure_en=True, pile_set_names=['PubMed Abstracts'],
                                                       word_counts=[1_500_000_000], test=True)
    else:
        pile_sample_dicts = the_pile.sample_pile_texts(ensure_en=True, pile_set_names=['PubMed Abstracts'],
                                                       word_counts=[1_500_000_000])
    log_and_flush(f"Saving to {save_path}")
    save_to_huggingface_format(pile_sample_dicts, save_path)


if __name__ == "__main__":
    main()
