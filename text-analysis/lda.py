import numpy as np
import pandas as pd
import gensim
import pickle
import random
from scipy.stats import entropy
from numpy.linalg import norm
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

def JSD(P, Q):
	"""
	Jensen–Shannon divergence
	"""
	_P = P / norm(P, ord=1)
	_Q = Q / norm(Q, ord=1)
	_M = 0.5 * (_P + _Q)
	return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def get_similar_documents(query, doc_rep, k=5):
	similarity_arr = []
	for i in range(doc_rep.shape[0]):
		similarity_arr.append(JSD(query, doc_rep[i]))
	return np.array(similarity_arr).argsort()[:k]

def plot_doc_rep(doc_rep, save_folder):
	plt.figure(figsize = (12,8))
	plt.title('Document representation in terms of topic prob distribution')
	plt.xlabel('topic 1')
	plt.ylabel('topic 2')
	plt.grid()
	plt.plot(doc_rep[:,:1], doc_rep[:,1:2], 'ro')
	plt.savefig('{0}/lda_document_plot.png'.format(save_folder))
	plt.show()


random.seed(9001)
root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')
doc_term = 'doc_term_matrix_50000.pkl'
vocab_file = 'vocabulary_50000.pkl'
lda_model_file = 'lda.pkl'
topic_results = 'results.txt'
similar_doc_results = 'similar_doc.txt'
lda_doc_rep= 'doc_rep.pkl'
similarity_mat = 'sim_mat.pkl'
num_topics = 5
num_words = 10


def lda_analysis(root_folder):
	print('Loading document term matrix and vocabulary.....')
	# load the document term matrix and vocabulary
	with open('{0}/{1}'.format(root_folder, doc_term),'rb')as fp:
	    doc_term_matrix = pickle.load(fp) 
	with open('{0}/{1}'.format(root_folder, vocab_file),'rb')as fp:
	    vocab = pickle.load(fp)

	print('Building LDA model.....')
	# lda model
	lda_model = gensim.models.ldamodel.LdaModel(gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False), num_topics=num_topics, 
		id2word = {v: k for k, v in vocab.items()}, passes=50)

	with open('{0}/{1}'.format(root_folder, lda_model_file), 'wb') as fp:
		pickle.dump(lda_model, fp)

	print('Writing top 10 words in each topic.........')
	# top 10 words in each topic
	result_file = open('{0}/{1}'.format(root_folder,topic_results), 'w')
	for topicno in range(lda_model.num_topics):
		result_file.write('######### Topic : {0} ##########\n'.format(topicno + 1))
		for word, prob in lda_model.show_topic(topicno, topn=num_words):
			result_file.write('{0},{1}\n'.format(word, prob))
	result_file.close()

	print('Writing document topic distribution.........')
	# document representation
	corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
	doc_topic_dist = []
	for i in range(len(corpus)):
		dist = [prob for _, prob in lda_model.get_document_topics(corpus[i])]
		if len(dist) == num_topics:
			doc_topic_dist.append(np.array(dist))
	doc_topic_dist = np.array(doc_topic_dist)

	with open('{0}/{1}'.format(root_folder,lda_doc_rep), 'wb') as fp:
		pickle.dump(doc_topic_dist, fp)

	plot_doc_rep(doc_topic_dist, root_folder)

	# print('Finding similar documents for a given set of documents.....')
	# Since document is a probability distribution of topics, Jensen–Shannon divergence is used for measuring the similarity

	# print('Similarity matrix generation....')
	# # similarity matrix
	# sim_mat = pdist(doc_topic_dist, JSD)
	# with open('{0}/{1}'.format(root_folder,similarity_mat), 'wb') as fp:
	# 	pickle.dump(sim_mat, fp)

	# choose random 5 documents
	docs = list(range(len(corpus)))
	random.shuffle(docs)
	query_documents = docs[:5]

	print('Similarity results generation.....')
	sim_file = open('{0}/{1}'.format(root_folder,similar_doc_results), 'w')
	for query_doc in query_documents:
		query = doc_topic_dist[query_doc]
		sim_file.write('######### Document : {0} ##########\n'.format(query_doc))
		sim_file.write('{0}\n'.format(get_similar_documents(query, doc_topic_dist)))
	sim_file.close()

lda_analysis(folder_dataset_2)
lda_analysis(folder_dataset_1)




