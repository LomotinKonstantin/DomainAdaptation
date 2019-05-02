import re
from string import punctuation

from nltk.stem import SnowballStemmer
from nltk import word_tokenize


class Preprocessor:
    """
    Приведение в нижний регистр, удаление пунктуации и цифр
    """

    def __init__(self):
        self.to_drop = punctuation + "0123456789“"
        self.stemmer = SnowballStemmer("english")

    def _drop_symbols(self, text: str) -> str:
        for c in self.to_drop:
            text = text.replace(c, " ")
        return re.sub(r"\s{2,}]", " ", text)

    def preprocess(self, text: str) -> str:
        # Lowercase conversion
        return " ".join(self.preprocess_tokens(text))

    def preprocess_tokens(self, text: str) -> list:
        # Lowercase conversion
        new_text = text.lower()
        new_text = self._drop_symbols(new_text)
        tokens = word_tokenize(new_text)
        return [self.stemmer.stem(t) for t in tokens if len(t) > 1]
