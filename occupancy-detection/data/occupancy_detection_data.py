import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class OccupancyDetectionDataset(Dataset):
	'''
	Occupany Detection Dataset
	'''  

	def __init__(self, input_file, root_folder, predictors, target):
		columns = list(predictors)
		columns.append(target)
		self.data_frame = pd.read_csv('{0}/{1}'.format(root_folder, input_file))[columns]
		self.target = target
		self.predictors = predictors
		self.root_folder = root_folder

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		return { 'X' : self.data_frame[self.predictors].iloc[idx].values , 'Y' :  self.data_frame[self.target].iloc[idx].reshape(-1)}
