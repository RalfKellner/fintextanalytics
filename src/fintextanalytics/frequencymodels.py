import os
from .utils import read_word_list
from collections import Counter
import pandas as pd


class FreqAnalyzer:
    def __init__(self):
        self.wordlists = self._load_default_wordlists()


    def count_words_in_wordlist(self, token_list, wordlist = 'esgwords'):
        '''
        Count the occurence of words from a word list in a list of strings. The list of strings can be generated from raw text using the
        text_preprocessor function from the utils module. However, you are free to preprocess raw text into a list of tokens as you like.
        A few word lists are given by default and initialized with the class instance. You can also use your own list of words.

        Parameters:
        -----------
        token_list: list of str
        wordlist: str or list of str
            If you want to use internal word lists the string must be either: ewords, swords, gwords or esgwords. Or, you simply provide a list
            of strings, e.g., ['expect', 'gdp', 'revenue']

        Returns:
        ---------
        pd.DataFrame

        '''

        self._validate_words(wordlist)
        word_counts = Counter(token_list)
        word_counts_df = pd.DataFrame(data = word_counts.values(), index = word_counts.keys())            

        if isinstance(wordlist, str):
            available_words = [word for word in self.wordlists[wordlist] if word in list(word_counts_df.index)]
        else:
            available_words = [word for word in wordlist if word in list(word_counts_df.index)]

        return word_counts_df.loc[available_words]


    @staticmethod
    def _load_default_wordlists():
        '''
        This function is made for internal use of the class.
        '''
        
        this_dir, filename = os.path.split(__file__)
        data_dir = os.path.join(this_dir, 'data')
        
        word_list = dict()
        for filename in ["ewords.txt", "swords.txt", "gwords.txt"]:
            word_list[filename.split(".")[0]] = read_word_list(os.path.join(data_dir, filename))
        all_words = []
        
        for key in word_list.keys():
            all_words += word_list[key]
        word_list['esgwords'] = all_words
        
        return word_list
    
    @staticmethod
    def _validate_words(words):
        '''
        This function is made for internal use of the class.
        '''
        if isinstance(words, str):
            assert words in ['ewords', 'swords', 'gwords', 'esgwords'], 'When using internal word list, words must be "ewords", "swords", "gwords" or "esgwords"'
        elif isinstance(words, list):
            assert all([isinstance(word, str) for word in words]), 'If a user defined word list is provided, a list of strings must be used.'
        else:
            raise TypeError('Either use a string for build-in word lists or provide a list of strings.')

    


        
