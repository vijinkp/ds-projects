# https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
# https://github.com/mcleonard/pytorch-charRNN
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# https://www.kaggle.com/mikebaik/simple-rnn-with-pytorch
import glob, unicodedata, string
import torch
from torch.Autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers=self.n_layers, nonlinearity='tanh', batch_first=True)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        self.rnn.flatten_parameters()
        output, self.hidden = self.rnn(x, self.hidden)
        output = self.linear(output)
        output = F.softmax(output)
    	return output

    def init_hidden(self):
        return Variable(torch.randn(self.n_layers, 1, self.hidden_size), requires_grad=True)



# Extended ascii of 256 characters
vocab = [chr(char) for char in range(256)]
