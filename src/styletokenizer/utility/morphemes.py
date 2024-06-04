import nltk
import spacy
from nltk.stem import PorterStemmer

# Load the English language model
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')


# Function to tokenize text into morphemes
def morpheme_tokenizer_spacy(text):
    # Process the text with spaCy
    doc = nlp(text.lower())
    morphemes = []

    for token in doc:
        if len(token.prefix_) > 1:
            morphemes.append(token.prefix_)
        if token.lemma_ in token.text:
            morphemes.append(token.lemma_)
        if token.suffix_ != token.lemma_:
            morphemes.append(token.suffix_)

    return morphemes


def stem_word(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


# Example text
text = "The children are unhappiness and aren't playing in the garden."
text = "caresses"

# Tokenize text into morphemes
morphemes = stem_word(text)

# Print the results
print(morphemes)
