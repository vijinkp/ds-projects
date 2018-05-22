import numpy as np
from matplotlib import pyplot as plt
import pickle

def plot(title, xlabel, ylabel, X_list, Y_list, L_list, save_file):
	plt.figure(figsize=(12,8))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	legend_tup = []
	legend_label = []
	for i in range(len(X_list)):
		l, = plt.plot(X_list[i], Y_list[i])
		legend_tup.append(l)
		legend_label.append(L_list[i])
	plt.xticks(X_list[0], X_list[0])
	plt.grid()
	plt.legend(tuple(legend_tup), tuple(legend_label))
	plt.savefig(save_file)
	plt.show()



# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_train_loss_map.pkl', 'rb') as fp:
#     relu_train_loss_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_test_loss_map.pkl', 'rb') as fp:
#     relu_test_loss_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/sigmoid_20_train_loss_map.pkl', 'rb') as fp:
#     sig_train_loss_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/sigmoid_20_test_loss_map.pkl', 'rb') as fp:
#     sig_test_loss_map = pickle.load(fp)


# ReLU & Sigmoid
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/sigmoid_20_train_accuracy_map.pkl', 'rb') as fp:
#     sig_train_accuracy_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/sigmoid_20_test_accuracy_map.pkl', 'rb') as fp:
#     sig_test_accuracy_map = pickle.load(fp)

# # accuracy plot
# plot('Train accuracy using ReLU and sigmoid', '#epochs', 'train accuracy', 
#  	[list(relu_train_accuracy_map.keys()), list(sig_train_accuracy_map.keys())], 
#  	[list(relu_train_accuracy_map.values()), list(sig_train_accuracy_map.values())], 
#  	['ReLU', 'Sigmoid'], '/home/vparambath/Desktop/iith/ds-projects/data/mnist/train_accuracy_plot.png')

# plot('Test accuracy using ReLU and sigmoid', '#epochs', 'test accuracy', 
#  	[list(relu_test_accuracy_map.keys()), list(sig_test_accuracy_map.keys())] , 
#  	[list(relu_test_accuracy_map.values()), list(sig_test_accuracy_map.values())], 
#  	['ReLU', 'Sigmoid'], '/home/vparambath/Desktop/iith/ds-projects/data/mnist/test_accuracy_plot.png')


# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_25_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_50_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_50_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_75_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_75_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_100_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_100_map = pickle.load(fp)

# # train accuracy plot
# plot('Train accuracy for different Dropout probabilities', '#epochs', 'train accuracy', 
#  	[list(relu_train_accuracy_25_map.keys()), list(relu_train_accuracy_50_map.keys()), 
#  	list(relu_train_accuracy_75_map.keys()), list(relu_train_accuracy_100_map.keys())], 
#  	[list(relu_train_accuracy_25_map.values()), list(relu_train_accuracy_50_map.values()), 
#  	list(relu_train_accuracy_75_map.values()), list(relu_train_accuracy_100_map.values())], 
#  	['Dropout prob : 0.25', 'Dropout prob : 0.50', 'Dropout prob : 0.75', 'Dropout prob : 1.00'], 
#  	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/train_accuracy_dropout_plot.png')


# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_25_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_50_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_50_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_75_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_75_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_100_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_100_map = pickle.load(fp)

# # train accuracy plot
# plot('Test accuracy for different Dropout probabilities', '#epochs', 'test accuracy', 
#  	[list(relu_test_accuracy_25_map.keys()), list(relu_test_accuracy_50_map.keys()), 
#  	list(relu_test_accuracy_75_map.keys()), list(relu_test_accuracy_100_map.keys())], 
#  	[list(relu_test_accuracy_25_map.values()), list(relu_test_accuracy_50_map.values()), 
#  	list(relu_test_accuracy_75_map.values()), list(relu_test_accuracy_100_map.values())], 
#  	['Dropout prob : 0.25', 'Dropout prob : 0.50', 'Dropout prob : 0.75', 'Dropout prob : 1.00'], 
#  	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/test_accuracy_dropout_plot.png')


# batch normalization and dropout
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_bn_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_25_bn_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_bn_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_25_bn_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_bn_woDP_train_accuracy_map.pkl', 'rb') as fp:
#     relu_train_accuracy_25_bn_woDP_map = pickle.load(fp)
# with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_25_bn_woDP_test_accuracy_map.pkl', 'rb') as fp:
#     relu_test_accuracy_25_bn_woDP_map = pickle.load(fp)

# plot('Train accuracy for Batch Normalization with & without Dropout', '#epochs', 'train accuracy', 
#  	[list(relu_train_accuracy_25_bn_map.keys()), list(relu_train_accuracy_25_bn_woDP_map.keys())], 
#  	[list(relu_train_accuracy_25_bn_map.values()), list(relu_train_accuracy_25_bn_woDP_map.values())], 
#  	['Dropout', 'No Dropout'], 
#  	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/train_accuracy_bn_dropout_plot.png')

# plot('Test accuracy for Batch Normalization with & without Dropout', '#epochs', 'test accuracy', 
#  	[list(relu_test_accuracy_25_bn_map.keys()), list(relu_test_accuracy_25_bn_woDP_map.keys())], 
#  	[list(relu_test_accuracy_25_bn_map.values()), list(relu_test_accuracy_25_bn_woDP_map.values())], 
#  	['Dropout', 'No Dropout'], 
#  	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/test_accuracy_bn_dropout_plot.png')


# weight initialization comparison
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_uniform_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_uniform = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_kaiming_uniform_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_kaiming_uniform = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_kaiming_normal_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_kaiming_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_xavier_normal_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_xavier_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_xavier_uniform_train_accuracy_map.pkl', 'rb') as fp:
    train_accuracy_xavier_uniform = pickle.load(fp)

plot('Train accuracy for different weight initializations', '#epochs', 'train accuracy', 
 	[list(train_accuracy_normal.keys()), list(train_accuracy_uniform.keys()), list(train_accuracy_kaiming_uniform.keys()), 
 	list(train_accuracy_kaiming_normal.keys()), list(train_accuracy_xavier_normal.keys()), list(train_accuracy_xavier_uniform.keys())], 
 	[list(train_accuracy_normal.values()), list(train_accuracy_uniform.values()), list(train_accuracy_kaiming_uniform.values()), 
 	list(train_accuracy_kaiming_normal.values()), list(train_accuracy_xavier_normal.values()), list(train_accuracy_xavier_uniform.values())], 
 	['Normal', 'Uniform', 'Kaiming Uniform', 'Kaiming Normal', 'Xavier Normal', 'Xavier Uniform'], 
 	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/train_accuracy_weight_init_plot.png')

with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_uniform_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_uniform = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_kaiming_uniform_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_kaiming_uniform = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_kaiming_normal_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_kaiming_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_xavier_normal_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_xavier_normal = pickle.load(fp)
with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_xavier_uniform_test_accuracy_map.pkl', 'rb') as fp:
    test_accuracy_xavier_uniform = pickle.load(fp)

plot('Test accuracy for different weight initializations', '#epochs', 'test accuracy', 
 	[list(test_accuracy_normal.keys()), list(test_accuracy_uniform.keys()), list(test_accuracy_kaiming_uniform.keys()), 
 	list(test_accuracy_kaiming_normal.keys()), list(test_accuracy_xavier_normal.keys()), list(test_accuracy_xavier_uniform.keys())], 
 	[list(test_accuracy_normal.values()), list(test_accuracy_uniform.values()), list(test_accuracy_kaiming_uniform.values()), 
 	list(test_accuracy_kaiming_normal.values()), list(test_accuracy_xavier_normal.values()), list(test_accuracy_xavier_uniform.values())], 
 	['Normal', 'Uniform', 'Kaiming Uniform', 'Kaiming Normal', 'Xavier Normal', 'Xavier Uniform'], 
 	'/home/vparambath/Desktop/iith/ds-projects/data/mnist/test_accuracy_weight_init_plot.png')
