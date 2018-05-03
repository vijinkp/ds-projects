import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def generate_train_epoch_plot(train_error_data, learning_rate, base_rate, hidden_size):
	no_epochs = len(train_error_data)
	plt.figure(figsize=(12,8), facecolor='#DCDCDC')
	plt.title('Training error vs number of epochs (Learning rate: {0}, # Hidden units: {1})'.format(learning_rate, hidden_size))
	plt.xlabel('# epochs')
	plt.ylabel('training error')
	plt.xticks([i for i in range(no_epochs)], [i+1 for i in range(no_epochs)])
	plt.grid()
	plt.margins(0,0.05)
	l1, = plt.plot(range(no_epochs), [base_rate for i in range(no_epochs)], 'r-')
	l2, = plt.plot(range(no_epochs), train_error_data, 'bo-')
	plt.legend((l1,l2) , ('baseline model', 'MLP model'))
	plt.savefig('train_error.png')



def generate_hidden_size_perf_plot(hidden_size_list, train_perf_list, test_perf_list, base_rate, perf_metric='Accuracy'):
	hs_len = len(hidden_size_list)
	plt.figure(figsize=(12,8), facecolor='#DCDCDC')
	plt.title('Training & Testing performance({0}) for different hidden units'.format(perf_metric))
	plt.xlabel('# hidden units')
	plt.ylabel(perf_metric.lower())
	plt.xticks([i for i in range(hs_len)], hidden_size_list)
	plt.grid()
	plt.margins(0,0.05)
	l1, = plt.plot(range(hs_len), [base_rate for i in range(hs_len)], 'r-')
	l2, = plt.plot(range(hs_len), train_perf_list, 'bo-')
	l3, = plt.plot(range(hs_len), test_perf_list, 'go-')
	plt.legend((l1,l2,l3) , ('Baseline {0}'.format(perf_metric), 'Training {0}'.format(perf_metric), 'Testing {0}'.format(perf_metric)))
	plt.savefig('perf_{0}_hidden_unit.png'.format(perf_metric.lower()))


if __name__ == '__main__':


	base_auc = 0.50106320248
	base_accuracy = 0.501538146021

	hidden_size_list = [1, 2, 5, 10, 20]
	training_accuracy = [0.7876703917, 0.9225101314, 0.933562569078, 0.961193663269, 0.983175733759]
	training_auc = [0.5, 0.938770615104, 0.957192939173, 0.972409165679, 0.988052856831]
	testing_accuracy = [0.789889253486, 0.942268252666, 0.945447087777, 0.98656685808, 0.987489745693]
	testing_auc = [0.5, 0.962739349649, 0.964930664385, 0.990063925833, 0.988856996827]

	generate_hidden_size_perf_plot(hidden_size_list, training_accuracy, testing_accuracy, base_accuracy)
	generate_hidden_size_perf_plot(hidden_size_list, training_auc, testing_auc, base_auc, 'AUC')




