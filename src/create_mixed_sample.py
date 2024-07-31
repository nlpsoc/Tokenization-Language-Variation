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
import pandas as pd
from datasets import Dataset, concatenate_datasets
import argparse
import styletokenizer.utility.s2orc as s2orc
import styletokenizer.utility.sadiri as sadiri
import styletokenizer.utility.the_pile as the_pile
import styletokenizer.utility.youtube_commons as youtube_commons


def main(save_path='/shared/3/projects/hiatus/TOKENIZER_wegmann/data/fitting-corpora/mixed', test=False):
    if not test:
        s2orc_sample_dict = s2orc.sample_s2orc_texts()
        youtube_sample_dict = youtube_commons.sample_YouTubeCommons_texts()
        sadiri_sample_dict = sadiri.sample_sadiri_texts()
        pile_sample_dict = the_pile.sample_pile_texts()
    else:
        s2orc_sample_dict = s2orc.sample_s2orc_texts(required_word_count=10)
        youtube_sample_dict = youtube_commons.sample_YouTubeCommons_texts(required_word_count=10)
        sadiri_sample_dict = sadiri.sample_sadiri_texts(word_samples=[10 for _ in range(len(sadiri.SET_PATHS))])
        pile_sample_dict = the_pile.sample_pile_texts(sampled_word_counts=
                                                      [10 for _ in range(len(the_pile.PILE_SET_NAMES))])

    # Convert dictionaries to pandas DataFrames
    sadiri_df = pd.DataFrame.from_dict(sadiri_sample_dict)
    s2orc_df = pd.DataFrame.from_dict(s2orc_sample_dict)
    pile_df = pd.DataFrame.from_dict(pile_sample_dict)
    youtube_df = pd.DataFrame.from_dict(youtube_sample_dict)

    # Convert DataFrames to Hugging Face Datasets
    sadiri_dataset = Dataset.from_pandas(sadiri_df)
    s2orc_dataset = Dataset.from_pandas(s2orc_df)
    pile_dataset = Dataset.from_pandas(pile_df)
    youtube_dataset = Dataset.from_pandas(youtube_df)

    # Concatenate the datasets
    combined_dataset = concatenate_datasets([sadiri_dataset, s2orc_dataset, pile_dataset, youtube_dataset])

    # Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=42)

    # Save the combined dataset to the specified path
    shuffled_dataset.save_to_disk(save_path)

    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    # add boolean command line argument test
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run the script in test mode')
    args = parser.parse_args()
    main(test=args.test)
