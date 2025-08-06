import spacy

# Load English small model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Process the text with spaCy
    doc = nlp(text)
    # Keep only meaningful words (no stopwords, no punctuations, no numbers)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    # Join tokens back into a string
    return " ".join(tokens)
