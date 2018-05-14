import numpy as np
import pandas as pd
import pickle
import random
from matplotlib import pyplot as plt
from sklearn.utils.extmath import randomized_svd


def plot_svd_reconstruction_error(doc_term_matrix, save_folder, n_components=1500):
	"""
	The error between a matrix A and its rank-k approximation Ak has a (spectral) norm2 given by the k+1-th singular value of A.
	https://nlp.stanford.edu/IR-book/html/htmledition/low-rank-approximations-1.html
	https://stackoverflow.com/questions/28571399/matrix-low-rank-approximation-using-matlab
	"""
	U,S,VT = randomized_svd(doc_term_matrix, n_components=n_components, random_state = 9001)
	error = S[1:]
	plt.figure(figsize = (12,8))
	plt.title('SVD reconstruction error vs k')
	plt.xlabel('frobeious norm of error')
	plt.ylabel('k')
	plt.grid()
	plt.plot(range(len(error)), error, 'ro-')
	plt.savefig('{0}/frobeious norm_plot.png'.format(save_folder))
	plt.show()

random.seed(9001)
root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')
doc_term = 'doc_term_matrix_50000.pkl'
vocab_file = 'vocabulary_50000.pkl'
n_components = 1500


print('Loading document-term matrix and vocabulary.....')
# load the document term matrix and vocabulary
with open('{0}/{1}'.format(root_folder, doc_term),'rb')as fp:
    doc_term_matrix = pickle.load(fp) 
with open('{0}/{1}'.format(root_folder, vocab_file),'rb')as fp:
    vocab = pickle.load(fp)


# polysemy : absorb, acquire, admit, assume, claim, conclude, cut, deny, dictate, drive, edit, enjoy, fire, grasp, know, launch, explain, fall, lead, meet, go








