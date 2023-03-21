import pickle
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import os
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from sklearn.preprocessing import normalize
import numpy as np


def text_preprocessor(text):
    '''
    A function for preprocessing text. The function uses the simple_preprocess function in combination with the strip_tag function
    from the gensim package. It removes numbers, tags and other symbols and returns a list of tokens.

    Paramters:
    ----------
    text: str

    Returns:
    --------
    list of str
    
    '''
    return simple_preprocess(strip_tags(text), deacc=True)


def report2sentences(raw_text, idx_start=0, min_words=10, max_words=100, **kwargs):
    '''
    A function which takes a document, splits and preprocesses its content into sentences.

    Parameters:
    -----------
    raw_text: str
        a string which is usually a part of a report, e.g., item 1a from a 10-K report.

    idx_start: int
        the function output is a dataframe with an index serving as a sentence identifier, the id starts at this index number

    min_words: int
        some sentences maybe omitted if they are too short, sentences with a word count lower than this number are set to false for usage in
        the output
    
    max_words: int
        some sentences maybe omitted if they are too long, sentences with a word count higher than this number are set to false for usage in
        the output

    **kwargs: None
        you can specify arbitrary information which is added columnwise to the output, e.g., the company name of the report or date of publication

    Returns:
    ---------
    pd.DataFrame
    '''
    
    # identify unique sentences
    sentences = sent_detector.tokenize(raw_text.strip())
    # preprocess each sentence
    sentences_prep = [text_preprocessor(sentence) for sentence in sentences]
    # check if sentences should be used; very uncommon sentences in terms of length may not be representative
    sentences_use = [(min_words <= len(sentence_prep) <= max_words) for sentence_prep in sentences_prep]

    # joint each token list for saving in the data frame
    sentences_join = [','.join(sentence_prep) for sentence_prep in sentences_prep]
    
    # check is something odd happens during the first two steps
    assert len(sentences) == len(sentences_join), 'See if something odd occurs for text preparation, raw number of sentences is not equal to the number of preprocessed sentences.'
    
    # generate an index for the dataframe; if one iterates through many texts, it may be desired to build an index over all texts
    idx = list(range(idx_start, idx_start + len(sentences), 1))
    # generate a dataframe
    sentences_df = pd.DataFrame({'raw_text': sentences, 'prep_text': sentences_join, 'use_doc': sentences_use}, index = idx)
    
    # by using keywords, it is possible to add metadata of the text for later usage
    if kwargs:
        for key, value in kwargs.items():
            sentences_df.loc[:, key] = value

    return sentences_df

def scrape_10k_items(cik, accension_nbr, email): 

    '''
    This function calls the API from the SEC (https://www.sec.gov/) scrapes the 10K reports which is identified by the cik of the company
    and the report's accension number.

    Parameters:
    ------------
    cik: str
    accension_nrb: str
    email: str
        your email address which reveals your id to the SEC

    Returns:
    ---------
    pd.DataFrame:
        a pandas dataframe including raw text of the report items 1, 1a, 3, 7 and 7a

    '''
    headers =  {'User-Agent': email}
    url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accension_nbr.replace("-", "")}/{accension_nbr}.txt'
    # Get the HTML data from the 2018 10-K from Apple
    r = requests.get(url, headers=headers)
    raw_10k = r.text

    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    # Create 3 lists with the span idices for each regex

    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
    ### First filter will give us document tag start <end> and document tag end's <start> 
    ### We will use this to later grab content in between these tags
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K' 
    ### as section names
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

    document = {}

    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            document[doc_type] = raw_10k[doc_start:doc_end]


    # Write the regex
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|1B|3|7A|7|8)\.{0,1})|(ITEM\s(1A|1B|3|7A|7|8))')

    # Use finditer to math the regex
    matches = regex.finditer(document['10-K'])

    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    test_df.columns = ['item', 'start', 'end']
    test_df['item'] = test_df.item.str.lower()

    # Get rid of unnesesary charcters from the dataframe
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)

    # Drop duplicates
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)

    # Get Item 1a
    item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item1b']]
    # Get Item 1b
    item_1b_raw = document['10-K'][pos_dat['start'].loc['item1b']:pos_dat['start'].loc['item3']]
    # Get Item 3
    item_3_raw = document['10-K'][pos_dat['start'].loc['item3']:pos_dat['start'].loc['item7']]
    # Get Item 7
    item_7_raw = document['10-K'][pos_dat['start'].loc['item7']:pos_dat['start'].loc['item7a']]
    # Get Item 7a
    item_7a_raw = document['10-K'][pos_dat['start'].loc['item7a']:pos_dat['start'].loc['item8']]

    ### First convert the raw text we have to exrtacted to BeautifulSoup object 
    item_1a_content = BeautifulSoup(item_1a_raw, 'lxml').get_text(' ')
    item_1b_content = BeautifulSoup(item_1b_raw, 'lxml').get_text(' ')
    item_3_content = BeautifulSoup(item_3_raw, 'lxml').get_text(' ')
    item_7_content = BeautifulSoup(item_7_raw, 'lxml').get_text(' ')
    item_7a_content = BeautifulSoup(item_7a_raw, 'lxml').get_text(' ')

    pos_dat.loc['item1a', 'text'] = item_1a_content
    pos_dat.loc['item1b', 'text'] = item_1b_content
    pos_dat.loc['item3', 'text'] = item_3_content
    pos_dat.loc['item7', 'text'] = item_7_content
    pos_dat.loc['item7a', 'text'] = item_7a_content

    pos_dat.drop(['item8'], axis = 0, inplace = True)

    return pos_dat


def write_word_list(list_of_strings, filename):
    '''
    Write a list of words to a text file

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to write a text file

    Returns:
    ---------
    None  
    '''
    with open(filename, 'w') as file:
        for word in list_of_strings:
            file.write("%s\n" % word)


def read_word_list(filename):
    '''
    Read a txt file including a list of words.

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to read a text file

    Returns:
    ---------
    list of str  

    '''
    word_list = []
    with open(filename, 'r') as file:
        for word in file:
            word_list.append(word[:-1])
    return word_list

def write_pickle(filename, obj):
    '''
    Save an object as a pickle file.

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to read a text file

    Returns:
    ---------
    list of str  
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):#
    '''
    Read a pickle file.

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to read a text file

    Returns:
    ---------
    object
    '''
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def load_example_text():
    '''
    This function is just for the purpose of demonstration
    '''
    this_dir, filename = os.path.split(__file__)
    data_dir = os.path.join(this_dir, 'data', 'example_text.txt')
    with open(data_dir, 'r') as file:
        text = file.read()
    return text

def load_example_reports():
    '''
    This function is just for the purpose of demonstration
    '''
    this_dir, filename = os.path.split(__file__)
    data_dir = os.path.join(this_dir, 'data', 'apple_10k_reportitems.csv')
    apple_reports = pd.read_csv(data_dir, index_col=False)
    return apple_reports
    
        