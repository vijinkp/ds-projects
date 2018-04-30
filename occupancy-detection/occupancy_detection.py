import torch
import pandas as pd
from data import OccupancyDetectionDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class OccupancyDetectionNet(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(OccupancyDetectionNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()
		self.out_act = nn.Sigmoid()

		torch.nn.init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2))
		torch.nn.init.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2))

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.out_act(out)
		return out

root_folder = '/home/vparambath/Desktop/iith/AML-Assignment'
predictors = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
target_col = "Occupancy"
train_file = 'train_data.txt'
test_file = 'test_data.txt'
batch_size = 8
input_size = len(predictors)
output_size = 1
num_epochs = 6
hidden_size = 5

train_dataset = OccupancyDetectionDataset(train_file, root_folder, predictors, target_col)
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)

# initialize network
net = OccupancyDetectionNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
net.train()
for epoch in range(num_epochs):
 	for i, batch in enumerate(train_dataloader):
 		data, target = Variable(batch['X'].float()), Variable(batch['Y'].float())

		optimizer.zero_grad()
		output = net(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print ('Epoch [{0}/{1}], Step [{2}/{3}], Loss: {4}'.format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))

torch.save(net.state_dict(), 'occupancy_detnet.pkl')

# Testing
net.eval()
correct = 0
total = 0
test_dataset = OccupancyDetectionDataset(test_file, root_folder, predictors, target_col)
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

actual = []
predicted = []
for j, test_batch in enumerate(test_dataloader):
	data, target = Variable(test_batch['X'].float()), Variable(test_batch['Y'].float())
	output = net(data)

	actual = actual + list(target.numpy().reshape(-1))
	predicted = predicted + list(output.data.numpy().reshape(-1))

predicted = np.round(predicted)
pred_df = pd.DataFrame({'Actual': actual, 'Predict': predicted})
print(pred_df[pred_df.Actual == pred_df.Predict].shape[0]/ float(len(test_dataset)))

