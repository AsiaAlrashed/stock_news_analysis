import spacy
import re
import string

nlp = spacy.load("en_core_web_sm")


class TextPreprocessor:
    """Class for cleaning texts using spaCy."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        """
        Clean text with spaCy.
        - Convert text to lowercase
        - Remove numbers, punctuation, and common words
        - Use lemmatization
        """
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token) > 2]
        return " ".join(tokens)
