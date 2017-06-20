import os
import re
import pandas as pd
from sklearn.externals import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


class MifStemmer():
    def __init__(self):
        self.stemmer = SnowballStemmer(language='russian')
        self.token = RegexpTokenizer(r'\b[^\d\W]+\b')
        self.stop = stopwords.words('russian')
        self.stop += stopwords.words('english')
        self.stop += ['br', 'nbsp', 'shy', 'em', 'nobr', 'p', 'b']
        self.stop += ['c', 'cам', 'e']

    def stem(self, text):
        result = []
        for w in self.token.tokenize(text.lower()):
            w_stem = self.stemmer.stem(w)
            if w_stem not in self.stop:
                result.append(w_stem)

        return result

    def __call__(self, text):
        return self.stem(text)
