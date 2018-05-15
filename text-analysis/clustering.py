import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

def elbow_method(data, save_folder):
	cluster_range = range( 1, 20 )
	cluster_errors = []
	for num_clusters in cluster_range:
		clusters = KMeans(init='k-means++', n_clusters =num_clusters, random_state=896)
		clusters.fit(data)
		#cluster_labels = clusters.fit_predict(data)
		cluster_errors.append(clusters.inertia_)
		#cluster_errors.append(sum(np.min(cdist(data, clusters.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
		#cluster_errors.append(silhouette_score(data, cluster_labels))
	clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors })
	plt.figure(figsize=(12,8))
	plt.title('Elbow method')
	plt.xlabel('# clusters')
	plt.ylabel('inertia')
	plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, "bo-")
	plt.xticks(np.arange(1,20), np.arange(1,20))
	plt.grid()
	plt.savefig('{0}/elbow_method.png'.format(save_folder))
	plt.show()

root_folder = '/home/vijin/iith/ds-projects/data/text-analysis'
folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')
doc_term = 'doc_term_matrix_50000.pkl'
vocab_file = 'vocabulary_50000.pkl'
top_words = 'top_words.txt'


def cluster_analysis(root_folder, no_dims = 2, no_clusters = 6):
	print('Loading document-term matrix and vocabulary.....')
	# load the document term matrix and vocabulary
	with open('{0}/{1}'.format(root_folder, doc_term),'rb')as fp:
		doc_term_matrix = pickle.load(fp) 
	with open('{0}/{1}'.format(root_folder, vocab_file),'rb')as fp:
		vocab = pickle.load(fp)
	inverted_vocab = {v: k for k, v in vocab.items()}

	print('Dimension reduction by Truncated SVD.....')
	svd = TruncatedSVD(n_components=no_dims, n_iter=7, random_state=42)
	reduced_matrix = svd.fit_transform(doc_term_matrix.T)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(reduced_matrix)
	
	#print('Elbow method.....')
	#elbow_method(X_scaled, root_folder)

	print('Clustering started.....')
	kmeans = KMeans(init='k-means++', n_clusters=no_clusters, random_state=124)
	kmeans.fit(X_scaled)

	print('Finding top 5 words in clusters...')
	top_words_file = open('{0}/{1}'.format(root_folder,top_words), 'w')
	for i,center in enumerate(kmeans.cluster_centers_):
	    dist = euclidean_distances(X_scaled , center.reshape(-1, 2))
	    top_words_file.write('######### Cluster : {0} ##########\n'.format(i+1))
	    top_words_file.write('{0}\n'.format([inverted_vocab[x] for x in dist.flatten().argsort()[:5]]))
	top_words_file.close()

# number of clusters are identified by elbow method
cluster_analysis(folder_dataset_1, no_clusters=6)
cluster_analysis(folder_dataset_2, no_clusters=7)




