"""

    SOURCE			GENRE					DOMAIN 								TOTAL 				 SAMPLE


    SADIRI			forum					reddit	 						  446,769,021 		   249,000,000
    SADIRI			literature				ao3					  			  573,926,907 		   100,000,000
    SADIRI			literature				BookCorpus2 					   57,367,225 			50,000,000
    ThePile			literature 				Gutenberg before 1919 					?				50,000,000
    SADIRI			news					realnews						  272,933,709 		   169,000,000
    SADIRI			news/comments			nytimes-articles-and-comments:     24,131,163 			24,131,163
    SADIRI			news/comments			sfu-socc 							3,007,117 			 3,007,117
    ThePile			Q&A 					stackexchange 				  	  		? 		   	   200,000,000
    SADIRI			reviews					goodreads 					  	   53,683,977  			53,683,977
    SADIRI			reviews					amazon	 						   31,650,279 			31,650,279
    SADIRI			mails					gmane 						      141,837,101  		   141,837,101
    YouTubeCommons	transcripts 			YouTubeCommons 					  	  				   100,000,000
    ThePile			transcripts				OpenSubtitles 											50,000,000
    ThePile			code				 	Github													50,000,000
    s2orc			science					s2orc					  	    				       100,000,000
    SADIRI			blogs					blogcorpus	 						 			 		 8,189,607
    ThePile			raw text webpages 		CommonCrawl 								    	   100,000,000
    ThePile			Mathematics 			DM Mathematics  										20,000,000
                                                                            --------------- 	---------------
                                            TOTAL: 						   		 				 1,500,499,244

"""
import argparse
import styletokenizer.utility.bookcorpus as bookcorpus
import styletokenizer.utility.s2orc as s2orc
import styletokenizer.utility.sadiri as sadiri
import styletokenizer.utility.the_pile as the_pile
import styletokenizer.utility.youtube_commons as youtube_commons
from styletokenizer.fitting_corpora import CORPORA_MIXED
from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from styletokenizer.utility.custom_logger import log_and_flush


def main(save_path=CORPORA_MIXED, test=False):
    log_and_flush("Creating mixed dataset")
    if not test:
        log_and_flush("Sampling from sadiri")
        sadiri_sample_dicts = sadiri.sample_sadiri_texts()
        log_and_flush("Sampling from the pile")
        pile_sample_dicts = the_pile.sample_pile_texts()
        log_and_flush("Sampling from s2orc")
        s2orc_sample_dicts = s2orc.sample_s2orc_texts()
        log_and_flush("Sampling from YouTubeCommons")
        youtube_sample_dicts = youtube_commons.sample_YouTubeCommons_texts()
    else:
        log_and_flush("Running in test mode")
        save_path = '/shared/3/projects/hiatus/TOKENIZER_wegmann/data/fitting-corpora/mixed_test'
        log_and_flush(f"Saving to {save_path}")

        # log_and_flush("Sampling from bookcorpus")
        # bookcorpus_sample_dicts = bookcorpus.sample_bookcorpus_texts(test=True)

        log_and_flush("Sampling from sadiri")
        sadiri_sample_dicts = sadiri.sample_sadiri_texts(test=True)
        log_and_flush(sadiri_sample_dicts)
        log_and_flush("Sampling from the pile")
        pile_sample_dicts = the_pile.sample_pile_texts(test=True)
        log_and_flush(pile_sample_dicts)
        log_and_flush("Sampling from s2orc")
        s2orc_sample_dicts = s2orc.sample_s2orc_texts(test=True)
        log_and_flush(s2orc_sample_dicts)
        log_and_flush("Sampling from YouTubeCommons")
        youtube_sample_dicts = youtube_commons.sample_YouTubeCommons_texts(test=True)
        log_and_flush(youtube_sample_dicts)

    # combine list of dicts into a single list
    all_dicts_list = s2orc_sample_dicts + youtube_sample_dicts + sadiri_sample_dicts + pile_sample_dicts
    if test:
        log_and_flush(all_dicts_list)

    # shuffle the list of dicts
    import random
    random.seed(42)
    random.shuffle(all_dicts_list)

    save_to_huggingface_format(all_dicts_list, save_path)

    # Convert dictionaries to pandas DataFrames
    # sadiri_df = pd.DataFrame.from_dict(sadiri_sample_dict)
    # s2orc_df = pd.DataFrame.from_dict(s2orc_sample_dict)
    # pile_df = pd.DataFrame.from_dict(pile_sample_dict)
    # youtube_df = pd.DataFrame.from_dict(youtube_sample_dict)

    # Convert DataFrames to Hugging Face Datasets
    # sadiri_dataset = Dataset.from_pandas(sadiri_df)
    # s2orc_dataset = Dataset.from_pandas(s2orc_df)
    # pile_dataset = Dataset.from_pandas(pile_df)
    # youtube_dataset = Dataset.from_pandas(youtube_df)

    # Concatenate the datasets
    # combined_dataset = concatenate_datasets([sadiri_dataset, s2orc_dataset, pile_dataset, youtube_dataset])

    # Shuffle the combined dataset
    # shuffled_dataset = combined_dataset.shuffle(seed=42)

    # Save the combined dataset to the specified path
    # shuffled_dataset.save_to_disk(save_path)

    # log_and_flush(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    # add boolean command line argument test
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run the script in test mode')
    args = parser.parse_args()
    main(test=args.test)
