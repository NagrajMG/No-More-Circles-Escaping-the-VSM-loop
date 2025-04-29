# Add your import statements here
import nltk
import re
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt
import math
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
# Add any utility functions here

delimiter = r'[.?!]'
subword_connectors = r"[-_/',]"
whitespaces = r'\s+'

# Function to map POS tags to WordNet POS tags
def pos_mapping(tag):
	if tag.startswith('J'): #if the token is an adjective
		return wordnet.ADJ
	elif tag.startswith('V'): #if the token is a verb
		return wordnet.VERB
	elif tag.startswith('N'): #if the token is a noun
		return wordnet.NOUN
	elif tag.startswith('R'): #if the token is an adverb
		return wordnet.ADV
	else:          
		return None
			
# Function to remove stopwords from a list of tokens
def stopwords_sieve(tokens , stopwords_list):
	return [token for token in tokens if token.lower() not in stopwords_list]


#find the highest ASCII size for the characters present in the queries.
def maxASCII(): 
    query_json = json.load(open("./cranfield/cran_queries.json", 'r'))[:]
    queries=[doc['query'] for doc in query_json]
    #flattening the list of queries in one list
    List = [c for query in queries for c in query] 
    vocab_size = -1
    # Loop through the list of characters and find the maximum ASCII size
    for char in np.unique(List):
        vocab_size=max(vocab_size, ord(char)+1)
    return vocab_size

