import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle
from scipy.linalg import svd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

def tokenize(data, stopwords = None):
	# Lower case
	data = data.lower()

	# Punctation removal
	data = data.translate(str.maketrans('', '',  string.punctuation))

	# Tokenization
	tokenizer = TreebankWordTokenizer()
	token_list = tokenizer.tokenize(data)
	
	# Removing stopwords
	if stopwords is not None:
		token_list = [word for word in token_list if word not in stopwords]

	# Stemming
	# stemmer = PorterStemmer()
	# token_list = [stemmer.stem(word) for word in token_list]

	# Lemmatization
	lemmatizer = WordNetLemmatizer()
	token_list = [lemmatizer.lemmatize(word) for word in token_list]

	# remove digits and single letter
	token_list = [word.strip() for word in token_list if len(word.strip()) > 1]

	return token_list


def generate_document_term_matrix(data, root_folder, data_name, stop_words, k=50000):

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
	with open('{0}/doc_term_matrix_{1}.pkl'.format(root_folder, k), 'wb') as fp:
		pickle.dump(doc_term_sparse_mat, fp)

	# saving vocabulary
	with open('{0}/vocabulary_{1}.pkl'.format(root_folder, k), 'wb') as fp:
		pickle.dump(tfidf_vectorizer.vocabulary_, fp) 

	print('Finished generating document term matrix for {0}....'.format(data_name))
	return doc_term_sparse_mat


def plot_document_mat(mat , data_name, save_folder):
    plt.figure(figsize = (15,12))
    plt.title('Data representation in reduced dimension: {0}'.format(data_name))
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.grid()
    plt.plot(mat[:, :1], mat[:, 1:], 'ro')
    for i, txt in enumerate(range(mat.shape[0])):
        plt.annotate(txt, (mat[:, :1][i],mat[:, 1:][i]))
    plt.savefig('{0}/{1}_dataplot.png'.format(save_folder, data_name))
    #plt.show()

def svd(data, dim=2):
	svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
	return svd.fit_transform(doc_term_matrix_d1)

# main intializations
english_stops = set(stopwords.words('english'))
root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
data_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'

folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')

# Read data
dataset1 = pd.read_csv('{0}/Dataset-1.csv'.format(data_folder))
dataset2 = pd.read_csv('{0}/Dataset-2.txt'.format(data_folder), sep=':', header=None, names=['TextId', 'Text'])

doc_term_matrix_d1 = generate_document_term_matrix(dataset1, folder_dataset_1, 'dataset1', english_stops)
doc_term_matrix_d2 = generate_document_term_matrix(dataset2, folder_dataset_2, 'dataset2', english_stops)

# document term reduced 2 dimensions
# with open('{0}/reduced2_document_matrix.pkl'.format(folder_dataset_1), 'wb') as fp:
# 		pickle.dump(svd(doc_term_matrix_d1), fp)

# with open('{0}/reduced2_document_matrix.pkl'.format(folder_dataset_2), 'wb') as fp:
# 		pickle.dump(svd(doc_term_matrix_d2), fp)

# # term document reduced 2 dimensions
# with open('{0}/reduced2_term_matrix.pkl'.format(folder_dataset_1), 'wb') as fp:
# 		pickle.dump(svd(doc_term_matrix_d1.T), fp)

# with open('{0}/reduced2_term_matrix.pkl'.format(folder_dataset_2), 'wb') as fp:
# 		pickle.dump(svd(doc_term_matrix_d2.T), fp)


# document represenation 


