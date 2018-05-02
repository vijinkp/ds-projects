import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def generate_train_epoch_plot(train_error_data, learning_rate, base_rate):
	no_epochs = len(train_error_data)
	plt.figure(figsize=(12,8), facecolor='#DCDCDC')
	plt.title('Training error vs number of epochs (Learning rate: {0})'.format(learning_rate))
	plt.xlabel('# epochs')
	plt.ylabel('training error')
	plt.xticks([i for i in range(no_epochs)], [i+1 for i in range(no_epochs)])
	plt.grid()
	plt.margins(0,0.05)
	l1, = plt.plot(range(no_epochs), [base_rate for i in range(no_epochs)], 'r-')
	l2, = plt.plot(range(no_epochs), train_error_data, 'bo-')
	plt.legend((l1,l2) , ('baseline model', 'MLP model'))
	plt.savefig('train_error.png')


