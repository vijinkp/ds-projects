import torch
import pandas as pd
from data import OccupancyDetectionDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score

from utils import generate_train_epoch_plot, get_baseline_performance

torch.manual_seed(752)

class OccupancyDetectionNet(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(OccupancyDetectionNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()
		self.out_act = nn.Sigmoid()

		# torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1)
		# torch.nn.init.constant(self.fc1.bias, 0.1)
		# torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1)
		# torch.nn.init.constant(self.fc2.bias, 0.1)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.out_act(out)
		return out

def percentage_accuracy(actuals, predicted):
	return accuracy_score(actuals, predicted)

root_folder = '/home/vparambath/Desktop/iith/AML-Assignment'
predictors = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
target_col = "Occupancy"
train_file = 'train_data.txt'
test_file = 'test_data.txt'
batch_size = 100
input_size = len(predictors)
output_size = 1
num_epochs = 20
hidden_size = 5
lr = 0.001
#lr_gamma = 0.55

# Baseline performance
base_accuracy, base_auc = get_baseline_performance(root_folder, test_file, target_col)

# Data
train_dataset = OccupancyDetectionDataset(train_file, root_folder, predictors, target_col)
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)

print('Training data size : {0}'.format(len(train_dataset)))

# initialize network
net = OccupancyDetectionNet(input_size, hidden_size, output_size)
print(net)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)#, momentum=0.8)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

loss_data = []
train_error_data = []
# Training
for epoch in range(num_epochs):
	net.train()
 	for i, batch in enumerate(train_dataloader):
 		data, target = Variable(batch['X'].float()), Variable(batch['Y'].float())
		optimizer.zero_grad()
		output = net(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		#scheduler.step()

		loss_data.append(loss.data.item())

		if (i+1) % 10 == 0:
			print ('Epoch [{0}/{1}], Step [{2}/{3}], Loss: {4}'.format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))

	# Training error at each epoch
	net.eval()
	train_actual = []
	train_predicted = []
	for j, train_batch in enumerate(train_dataloader):
		train_data, train_target = Variable(train_batch['X'].float()), Variable(train_batch['Y'].float())
		train_output = net(train_data)

		train_actual = train_actual + list(train_target.numpy().reshape(-1))
		train_predicted = train_predicted + list(train_output.data.numpy().reshape(-1))
	train_predicted = np.round(train_predicted)
	train_auc = roc_auc_score(train_actual, train_predicted)
	train_accuracy = percentage_accuracy(train_actual, train_predicted)
	train_error_data.append(1 - train_accuracy)
	print('Training accuracy : {0}'.format(train_accuracy))
	print('Training AUC : {0}'.format(train_auc))

#print('Training error : {0}'.format(train_error_data))
torch.save(net.state_dict(), 'occupancy_detnet.pkl')

# Save loss data @ batch level
with open('loss_data.pkl', 'wb') as fp:
	pickle.dump(loss_data, fp)

# Save train_error data @ epoch level
with open('train_error_data.pkl', 'wb') as fp1:
	pickle.dump(train_error_data, fp1)

# plots
generate_train_epoch_plot(train_error_data, lr, base_accuracy, hidden_size)

# Training error
net.eval()
train_actual = []
train_predicted = []
for j, train_batch in enumerate(train_dataloader):
	train_data, train_target = Variable(train_batch['X'].float()), Variable(train_batch['Y'].float())
	train_output = net(train_data)

	train_actual = train_actual + list(train_target.numpy().reshape(-1))
	train_predicted = train_predicted + list(train_output.data.numpy().reshape(-1))

precison, recall, thresholds = precision_recall_curve(train_actual, train_predicted)
train_predicted = np.round(train_predicted)
#print('Training Precision : {0}, Recall : {1}, Thresholds : {2}'.format(precison, recall, thresholds))
print('***************************************************************************************')
print('Overall Training AUC : {0}'.format(roc_auc_score(train_actual, train_predicted)))
print('Overall Training Accuracy : {0}'.format(percentage_accuracy(train_actual, train_predicted)))


# Testing
net.eval()
test_dataset = OccupancyDetectionDataset(test_file, root_folder, predictors, target_col)
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

actual = []
predicted = []
for j, test_batch in enumerate(test_dataloader):
	data, target = Variable(test_batch['X'].float()), Variable(test_batch['Y'].float())
	output = net(data)

	actual = actual + list(target.numpy().reshape(-1))
	predicted = predicted + list(output.data.numpy().reshape(-1))

precison, recall, thresholds = precision_recall_curve(actual, predicted)
predicted = np.round(predicted)
#print('Testing Precision : {0}, Recall : {1}, Thresholds : {2}'.format(precison, recall, thresholds))
print('Testing AUC : {0}'.format(roc_auc_score(actual, predicted)))
print('Testing Accuracy : {0}'.format(percentage_accuracy(actual, predicted)))

