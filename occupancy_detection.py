import torch
import pandas as pd
from data import OccupancyDetectionDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.Functional as F


class OccupancyDetectionNet(nn.Module):

	def __init__(self):


	def forward(self, x):


root_folder = '/home/vparambath/Desktop/iith/AML-Assignment'
predictors = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
target = "Occupancy"
train_file = 'train_data.txt'
test_file = 'test_data.txt'
batch_size = 16
input_size = len(predictors)
output_size = 1
num_epochs = 1
hidden_size = 5


dataset = OccupancyDetectionDataset('train_data.txt', root_folder, predictors, target)
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=4)

for i, batch in enumerate(dataloader):
	print(batch['X'])
	print('**********************')
	print(batch['Y'])