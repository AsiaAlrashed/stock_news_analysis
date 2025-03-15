import random
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet")


class DataAugmentor:
    """
    A class to improve data by replacing words with their synonyms.
    """

    def __init__(self):
        pass

    def synonym_replacement(self, sentence, n=1):
        """
        Replace words with their synonyms from WordNet.
        """
        words = sentence.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))

        if not random_word_list:
            return sentence

        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = wordnet.synsets(random_word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                new_words = [
                    synonym if word == random_word else word for word in new_words
                ]
                num_replaced += 1
            if num_replaced >= n:
                break

        return " ".join(new_words)

    def augment_text(self, text):
        """
        Adding improvements to texts through random synonyms.
        """
        if random.uniform(0, 1) < 0.3:
            text = self.synonym_replacement(text)
        return text
