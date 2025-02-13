"""
    file to create the misc training corpus (called mixed in the code)

    SOURCE			GENRE					DOMAIN 								TOTAL


    SADIRI			forum					reddit	 			 		   		250,000,000
    SADIRI			literature				ao3					   		   		150,000,000
    ThePile			literature 				Gutenberg before 1919 				 50,000,000
    SADIRI			news					realnews						  	150,000,000
    SADIRI			news/comments			nytimes-articles-and-comments:       25,000,000
    SADIRI			news/comments			sfu-socc 							  3,000,000
    ThePile			Q&A 					stackexchange 				  	  	200,000,000
    SADIRI			reviews					goodreads 					  	     50,000,000
    SADIRI			reviews					amazon	 						   	 50,000,000
    SADIRI			mails					gmane 						      	150,000,000
    YouTubeCommons	transcripts 			YouTubeCommons 					  	100,000,000
    ThePile			transcripts				OpenSubtitles 						 50,000,000
    ThePile			code				 	Github								 50,000,000
    s2orc			science					s2orc					  	    	100,000,000
    SADIRI			blogs					blogcorpus	 						 10,000,000
    ThePile			raw text webpages 		CommonCrawl 						100,000,000
    ThePile			Mathematics 			DM Mathematics  					 20,000,000
                                                                            ---------------
                                            TOTAL: 						   	  1,508,000,000

"""
import argparse
import styletokenizer.utility.s2orc as s2orc
import styletokenizer.utility.AO3 as sadiri
import styletokenizer.utility.the_pile as the_pile
import styletokenizer.utility.nytimes as nytimes
import styletokenizer.utility.amazon as amazon
import styletokenizer.utility.goodreads as goodreads
import styletokenizer.utility.gmane as gmane
import styletokenizer.utility.blogcorpus as blogcorpus
import styletokenizer.utility.realnews as realnews
import styletokenizer.utility.reddit as reddit
import styletokenizer.utility.sfu_socc as sfusocc
import styletokenizer.utility.youtube_commons as youtube_commons
from styletokenizer.fitting_corpora import CORPORA_MIXED
from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from styletokenizer.utility.custom_logger import log_and_flush


def main(save_path=CORPORA_MIXED, test=False):
    log_and_flush("Creating mixed dataset")
    if not test:
        log_and_flush(f"Saving to {save_path}")
        log_and_flush(f"Sampling from sfu-socc")
        sfu_socc_sample_dicts = sfusocc.sample_sfusocc_texts()
        log_and_flush(f"Sampling from reddit")
        reddit_sample_dicts = reddit.sample_reddit_texts()
        log_and_flush(f"Sampling from realnews")
        realnews_sample_dicts = realnews.sample_realnews_texts()
        log_and_flush(f"Sampling from blogcorpus")
        blogcorpus_sample_dicts = blogcorpus.sample_blogcorpus_texts()
        log_and_flush(f"Sampling from gmane")
        gmane_sample_dicts = gmane.sample_gmane_texts()
        log_and_flush(f"Sampling from goodreads")
        goodreads_sample_dicts = goodreads.sample_goodreads_texts()
        log_and_flush(f"Sampling from amazon")
        amazon_sample_dicts = amazon.sample_amazon_texts()
        log_and_flush(f"Sampling from nytimes-articles-and-comments")
        nytimes_sample_dicts = nytimes.sample_nytimes_texts()
        log_and_flush("Sampling from sadiri")
        sadiri_sample_dicts = sadiri.sample_sadiri_texts()
        log_and_flush("Sampling from the pile")
        pile_sample_dicts = the_pile.sample_pile_texts(ensure_en=True)
        log_and_flush("Sampling from s2orc")
        s2orc_sample_dicts = s2orc.sample_s2orc_texts()
        log_and_flush("Sampling from YouTubeCommons")
        youtube_sample_dicts = youtube_commons.sample_YouTubeCommons_texts()
    else:
        log_and_flush("Running in test mode")
        save_path = '/shared/3/projects/hiatus/TOKENIZER_wegmann/data/fitting-corpora/mixed_test'
        log_and_flush(f"Saving to {save_path}")

        log_and_flush("Sampling from sfu-socc")
        sfu_socc_sample_dicts = sfusocc.sample_sfusocc_texts(test=True)
        log_and_flush("Sampling from reddit")
        reddit_sample_dicts = reddit.sample_reddit_texts(test=True)
        log_and_flush("Sampling from realnews")
        realnews_sample_dicts = realnews.sample_realnews_texts(test=True)
        log_and_flush("Sampling from blogcorpus")
        blogcorpus_sample_dicts = blogcorpus.sample_blogcorpus_texts(test=True)
        log_and_flush("Sampling from gmane")
        gmane_sample_dicts = gmane.sample_gmane_texts(test=True)
        log_and_flush("Sampling from goodreads")
        goodreads_sample_dicts = goodreads.sample_goodreads_texts(test=True)
        log_and_flush("Sampling from amazon")
        amazon_sample_dicts = amazon.sample_amazon_texts(test=True)
        log_and_flush("Sampling from nytimes-articles-and-comments")
        nytimes_sample_dicts = nytimes.sample_nytimes_texts(test=True)
        log_and_flush("Sampling from sadiri")
        sadiri_sample_dicts = sadiri.sample_sadiri_texts(test=True)
        log_and_flush(sadiri_sample_dicts)
        log_and_flush("Sampling from the pile")
        pile_sample_dicts = the_pile.sample_pile_texts(ensure_en=True, test=True)
        log_and_flush(pile_sample_dicts)
        log_and_flush("Sampling from s2orc")
        s2orc_sample_dicts = s2orc.sample_s2orc_texts(test=True)
        log_and_flush(s2orc_sample_dicts)
        log_and_flush("Sampling from YouTubeCommons")
        youtube_sample_dicts = youtube_commons.sample_YouTubeCommons_texts(test=True)
        log_and_flush(youtube_sample_dicts)

    # combine list of dicts into a single list
    all_dicts_list = (s2orc_sample_dicts + youtube_sample_dicts + sadiri_sample_dicts + pile_sample_dicts +
                      nytimes_sample_dicts + amazon_sample_dicts + goodreads_sample_dicts + gmane_sample_dicts
                      + blogcorpus_sample_dicts + realnews_sample_dicts + reddit_sample_dicts + sfu_socc_sample_dicts)
    if test:
        log_and_flush(all_dicts_list)

    # shuffle the list of dicts
    import random
    random.seed(42)
    random.shuffle(all_dicts_list)

    save_to_huggingface_format(all_dicts_list, save_path)


if __name__ == "__main__":
    # add boolean command line argument test
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run the script in test mode')
    args = parser.parse_args()
    main(test=args.test)
