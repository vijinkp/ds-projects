import numpy as np
import pandas as pd
import pickle
import random
from matplotlib import pyplot as plt
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

root_folder = '/home/vijin/iith/ds-projects/data/text-analysis'
folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')
doc_term = 'doc_term_matrix_50000.pkl'
vocab_file = 'vocabulary_50000.pkl'
near_poly_words = 'near_poly_words.txt'
n_components = 1500
polysemous_words = ['absorb', 'acquire', 'admit', 'assume', 'claim', 'conclude', 'cut',
                  'deny', 'dictate', 'drive', 'edit', 'enjoy', 'fire', 'grasp', 'know', 
                  'launch', 'explain', 'fall', 'lead', 'meet', 'go']

def plot_svd_reconstruction_error(doc_term_matrix, save_folder, n_components=500):
	"""
	The error between a matrix A and its rank-k approximation Ak has a (spectral) norm2 given by the k+1-th singular value of A.
	https://nlp.stanford.edu/IR-book/html/htmledition/low-rank-approximations-1.html
	https://stackoverflow.com/questions/28571399/matrix-low-rank-approximation-using-matlab
	"""
	U,S,VT = randomized_svd(doc_term_matrix, n_components=n_components, random_state = 9001)
	error = S[1:]
	plt.figure(figsize = (12,8))
	plt.title('SVD reconstruction error vs k')
	plt.xlabel('k')
	plt.ylabel('frobeious norm of error')
	plt.grid()
	plt.plot(range(len(error)), error, 'bo-')
	plt.savefig('{0}/frobeious norm_plot.png'.format(save_folder))
	plt.show()


def visualize_representation(doc_term_matrix, vocabulary, save_folder, no_documents=100, no_terms=100, graph_type='both', seed=256):
	
	U, S, VT = svds(doc_term_matrix.T, k=2)
	inverted_vocab = {v: k for k, v in vocabulary.items()}

	# U represents terms and V represents document in term-document matrix 
	np.random.seed(seed)
	sampled_term_indices = np.random.choice(len(U), no_terms)
	sampled_doc_indices = np.random.choice(len(VT.T), no_documents)
	terms = U[sampled_term_indices]
	docs = VT.T[sampled_doc_indices]

	plt.figure(figsize = (15,12))
	plt.title('SVD representation')
	plt.xlabel('dim 1')
	plt.ylabel('dim 2')
	plt.grid()

	if graph_type == 'both':
		l1, = plt.plot(docs[:,:1], docs[:,1:2], 'rs')
		l2, = plt.plot(terms[:,:1], terms[:,1:2], 'go')
		for i, txt_index in enumerate(range(terms.shape[0])):
			plt.annotate(inverted_vocab[sampled_term_indices[i]], (terms[:, :1][i],terms[:, 1:][i]))
		for i, txt_index in enumerate(range(docs.shape[0])):
			plt.annotate(sampled_doc_indices[i], (docs[:, :1][i],docs[:, 1:][i]))
		plt.legend((l1,l2), ('documents', 'terms'))

	elif graph_type == 'document':
		l1, = plt.plot(docs[:,:1], docs[:,1:2], 'rs')
		for i, txt_index in enumerate(range(docs.shape[0])):
			plt.annotate(sampled_doc_indices[i], (docs[:, :1][i],docs[:, 1:][i]))
			plt.legend((l1), ('documents'))
	else:
		l2, = plt.plot(terms[:,:1], terms[:,1:2], 'go')
		for i, txt_index in enumerate(range(terms.shape[0])):
			plt.annotate(inverted_vocab[sampled_term_indices[i]], (terms[:, :1][i],terms[:, 1:][i]))
		plt.legend((l2), ('terms'))
	plt.savefig('{0}/svd_visual_plot.png'.format(save_folder))
	plt.show()

def nearest_polysemous_words(root_folder, polysemous_words, doc_term_matrix, vocabulary, no_components = 250, no_terms=5):
	inverted_vocab = {v: k for k, v in vocabulary.items()}
	word_indices = [vocabulary[x] for x in polysemous_words]
	svd = TruncatedSVD(n_components=no_components, n_iter=10, random_state=42)
	reduced_matrix = svd.fit_transform(doc_term_matrix.T)
	near_poly_words_file = open('{0}/{1}'.format(root_folder,near_poly_words), 'w')
	for i, index in enumerate(word_indices):
		near_poly_words_file.write('######### Polysemous word : {0} ##########\n'.format(polysemous_words[i]))
		near_poly_words_file.write('{0}\n'.format([inverted_vocab[x] for x in euclidean_distances(reduced_matrix, reduced_matrix[index].reshape(-1,no_components)).flatten().argsort()[1:6]]))
	near_poly_words_file.close()


def svd_analysis(root_folder):
	print('Loading document-term matrix and vocabulary.....')
	# load the document term matrix and vocabulary
	with open('{0}/{1}'.format(root_folder, doc_term),'rb')as fp:
	    doc_term_matrix = pickle.load(fp) 
	with open('{0}/{1}'.format(root_folder, vocab_file),'rb')as fp:
	    vocab = pickle.load(fp)

	print('Plotting SVD reconstruction error.....')
	plot_svd_reconstruction_error(doc_term_matrix, root_folder, n_components)

	print('Plotting SVD term document representation...')
	visualize_representation(doc_term_matrix, vocab, root_folder, 100, 100, 'both', 256)

	print('Finding nearest words....')
	nearest_polysemous_words(root_folder, polysemous_words, doc_term_matrix, vocab)

svd_analysis(folder_dataset_1)
svd_analysis(folder_dataset_2)







