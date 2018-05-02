import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import random

def generate_random_results(size):
	return np.random.randint(2, size=size)

def get_baseline_performance(root_folder, test_file, target, no_trails = 5):
	np.random.seed(123)

	# Read test file
	test_data = pd.read_csv('{0}/{1}'.format(root_folder, test_file))
	test_Y = test_data[target].values.reshape(-1)
	test_size = len(test_Y)

	# Generate random results
	full_r_results = []
	for i in range(no_trails):
		r_results = generate_random_results(test_size)
		full_r_results.append(r_results)
	full_r_results = np.array(full_r_results)
	pred = np.apply_along_axis(lambda col : np.argmax(np.bincount(col)), 0, full_r_results)

	accuracy = accuracy_score(test_Y, pred)
	auc = roc_auc_score(test_Y, pred)

	print('****Baseline random results****')
	print('AUC: {0}'.format(auc))
	print('Accuracy: {0}\n'.format(accuracy))

	return (accuracy,auc) 


def get_logistic_reg_performance(root_folder, train_file, test_file, predictors, target):

	train_data = pd.read_csv('{0}/{1}'.format(root_folder, train_file))
	test_data = pd.read_csv('{0}/{1}'.format(root_folder, test_file))

	train_X = train_data[predictors].values
	train_Y = train_data[target].values.reshape(-1)
	test_X = test_data[predictors].values
	test_Y = test_data[target].values.reshape(-1)

	logisticRegr = LogisticRegression()
	logisticRegr.fit(train_X, train_Y)
	pred = logisticRegr.predict(test_X)

	print('****Baseline Logistic regression results****')
	print('AUC: {0}'.format(roc_auc_score(test_Y, pred)))
	print('Accuracy: {0}\n'.format(accuracy_score(test_Y, pred)))

if __name__ == '__main__':
	root_folder = '/home/vparambath/Desktop/iith/AML-Assignment'
	predictors = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
	target_col = "Occupancy"
	train_file = 'train_data.txt'
	test_file = 'test_data.txt'

	get_logistic_reg_performance(root_folder, train_file, test_file, predictors, target_col)
	get_baseline_performance(root_folder, test_file, target_col)
	