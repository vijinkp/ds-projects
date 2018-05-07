# https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
import numpy as np
import pandas
import nltk


def get_tokens(data, stopwords = None):
	



root_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'
data_folder = '/home/vparambath/Desktop/iith/IR-Assignment2'

dataset_1 = pd.read_csv('{0}/Dataset-1.csv')
dataset_2 = pd.read_csv('{0}/Dataset-2.txt'.format(data_folder), sep=':', header=None, names=['TextId', 'Text'])
