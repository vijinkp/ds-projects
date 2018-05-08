# https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def tokenize(data, stopwords = None):
	# Lower case
	data = data.lower()
	data = unicode(data, "utf-8")

	# Punctation removal
	data = data.translate(None, string.punctuation)

	# Tokenization
	tokenizer = TreebankWordTokenizer()
	token_list = tokenizer.tokenize(data)
	
	# Removing stopwords
	if stopwords is not None:
		token_list = [word for word in token_list if word not in stopwords]

	# Stemming
	stemmer = PorterStemmer()
	token_list = [stemmer.stem(word) for word in token_list]
	return token_list


# main intializations
stemmer = PorterStemmer()
english_stops = set(stopwords.words('english'))
root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
data_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'


# Read data
#dataset_1 = pd.read_csv('{0}/Dataset-1.csv')
dataset_2 = pd.read_csv('{0}/Dataset-2.txt'.format(data_folder), sep=':', header=None, names=['TextId', 'Text'])

# Fixed vocabulary selection
# vocabulary size. Choose top k words from corpus after stopword removal and stemming
k = 5000
token_count_map = {}

# dataset 2
for text in dataset_2.Text:
	token_list = tokenize(text, english_stops)
	for token in token_list:
		if token in token_count_map:
			token_count_map[token] = token_count_map[token] + 1
		else:
			token_count_map[token] = 1

# sort token_count_map decreasing order of count
sorted_item_list = sorted(token_count_map.items(), key=lambda t: t[1], reverse=True)
vocabulary = set([x[0] for x in sorted_item_list[0:k]])

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=1, analyzer="word", stop_words=english_stops, vocabulary = vocabulary)
