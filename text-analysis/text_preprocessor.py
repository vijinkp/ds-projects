# https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle
from scipy.linalg import svd
from matplotlib import pyplot as plt
import seaborn as sns

def tokenize(data, stopwords = None):
	# Lower case
	data = data.lower()

	# Punctation removal
	data = data.translate(None, string.punctuation) # changes to be done for python 3

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


def generate_document_term_matrix(data, root_folder, data_name, stop_words, k=5000):

	print('Generating document term matrix for {0}....'.format(data_name))
	token_count_map = {}

	# vocabulary of k words based on frequency after stopword word removal, punctuation aremoval and stemming 
	for text in data.Text:
		token_list = tokenize(text, stop_words)
		for token in token_list:
			if token in token_count_map:
				token_count_map[token] = token_count_map[token] + 1
			else:
				token_count_map[token] = 1

	# sort token_count_map decreasing order of count
	sorted_item_list = sorted(token_count_map.items(), key=lambda t: t[1], reverse=True)
	vocabulary = set([x[0] for x in sorted_item_list[0:k]])

	tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=1, analyzer="word", stop_words=english_stops, vocabulary = vocabulary)
	doc_term_sparse_mat = tfidf_vectorizer.fit_transform(data.Text)

	# saving document term matrix
	with open('{0}/{1}_doc_term_matrix_{2}.pkl'.format(root_folder,data_name, k), 'wb') as fp:
		pickle.dump(doc_term_sparse_mat, fp)

	print('Finished generating document term matrix for {0}....'.format(data_name))

	return doc_term_sparse_mat


def plot_document_mat(mat , data_name, save_folder):
	plt.figure(figsize(12,12))
	plt.title('Data representation in reduced dimension: {0}'.format(data_name))
	plt.xlabel('dim 1')
	plt.ylabel('dim 2')
	plt.grid()
	plt.plot()
	plt.savefig('{0}/{1}_dataplot.png'.format(save_folder, data_name))



# main intializations
english_stops = set(stopwords.words('english'))
root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
data_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'


# Read data
# dataset_1 = pd.read_csv('{0}/Dataset-1.csv'.format(data_folder))
dataset_2 = pd.read_csv('{0}/Dataset-2.txt'.format(data_folder), sep=':', header=None, names=['TextId', 'Text'], nrows =250)

#generate_document_term_matrix(dataset_1, root_folder, 'dataset1', english_stops)
sparse_doc_mat_data2 = generate_document_term_matrix(dataset_2, root_folder, 'dataset2', english_stops)
doc_term_data2 = sparse_doc_mat_data2.toarray()

print('document term matrix shape :{0}'.format(doc_term_data2.shape))

# SVD
U, s, VT = svd(doc_term_data2)

# dimension reduction
no_dim = 2
sigma = zeros((doc_term_data2.shape[0], doc_term_data2.shape[1]))
sigma[:doc_term_data2.shape[0], :doc_term_data2.shape[0]] = diag(s)
sigma = sigma[:, : no_dim]

reduced_mat = U.dot(sigma)
print('reduced matrix shape : {0}'.format(reduced_mat.shape))







