import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# Define a function to POS tag a list of sentences
def tag(text):
    doc = nlp(text)
    return [token.pos_ for token in doc]
