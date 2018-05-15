import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

def elbow_method(data, save_folder):
	cluster_range = range( 2, 20 )
	cluster_errors = []
	for num_clusters in cluster_range:
		clusters = KMeans(init='k-means++', n_clusters =num_clusters, random_state=124)
		#clusters.fit(data)
		cluster_labels = clusters.fit_predict(data)
		#cluster_errors.append(clusters.inertia_)
		#cluster_errors.append(sum(np.min(cdist(data, clusters.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
		cluster_errors.append(silhouette_score(data, cluster_labels))
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

root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
folder_dataset_1 = '{0}/{1}'.format(root_folder, 'dataset1')
folder_dataset_2 = '{0}/{1}'.format(root_folder, 'dataset2')
doc_term = 'doc_term_matrix_50000.pkl'
vocab_file = 'vocabulary_50000.pkl'


def clustering(root_folder, no_dims = 2):
	print('Loading document-term matrix and vocabulary.....')
	# load the document term matrix and vocabulary
	with open('{0}/{1}'.format(root_folder, doc_term),'rb')as fp:
		doc_term_matrix = pickle.load(fp) 
	with open('{0}/{1}'.format(root_folder, vocab_file),'rb')as fp:
		vocab = pickle.load(fp)

	print('Dimension reduction by Truncated SVD.....')
	svd = TruncatedSVD(n_components=no_dims, n_iter=7, random_state=42)
	reduced_matrix = svd.fit_transform(doc_term_matrix.T)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(reduced_matrix)
	
	print('Elbow method.....')
	elbow_method(X_scaled, root_folder)


clustering(folder_dataset_1)
clustering(folder_dataset_2)




