# https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
# https://github.com/mcleonard/pytorch-charRNN
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# https://www.kaggle.com/mikebaik/simple-rnn-with-pytorch
# https://github.com/pytorch/examples/tree/master/word_language_model
import glob, unicodedata, string
import torch
from torch.autograd import Variable
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
		self.linear = nn.Linear(self.hidden_size, self.output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	
	def forward(self, input, hidden):
		input = input.unsqueeze(0)
		output, hidden = self.rnn(input.view(1,1,-1), hidden)
		output = self.linear(output.view(1,-1))
		output = self.softmax(output)
		return output, hidden

	def init_hidden(self):
		return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

def make_chars_input(chars, vocab):
    tensor = torch.zeros(len(chars), 1, len(vocab))
    for ci in range(len(chars)):
        char = chars[ci]
        tensor[ci][0][vocab.find(char)] = 1
    return Variable(tensor)

def make_target(chars, vocab):
    letter_indexes = [vocab.find(chars[li]) for li in range(len(chars))]
    tensor = torch.LongTensor(letter_indexes)
    return Variable(tensor)

# Extended ascii of 256 characters
vocab = ''.join([chr(char) for char in range(256)])
root_data_folder = '/home/vparambath/Desktop/iith/ds-projects/data/rnn/test_data'
sequence_len = 200
no_epochs = 1
epochs_evalute = [20, 40, 60, 80, 100]
hidden_size = 100
lr = 0.001


# corpus creation
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in vocab)

text_corpus = []
for file in glob.glob('{0}/*.txt'.format(root_data_folder)):
	with open(file, 'r') as fp:
		text_corpus.append(unicode_to_ascii(fp.read()))
text_corpus = ' '.join(text_corpus)

# initilaise network
network = RNN(len(vocab), hidden_size, len(vocab))
print(network)

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
criterion = nn.MSELoss()

# train network
total_iter = int(len(text_corpus)/sequence_len)
start_iter = 0
epoch_loss = []
for i in range(no_epochs):
	iter_loss = []
	avg_loss = 0
	#network.zero_grad()
	optimizer.zero_grad()
	while start_iter < len(text_corpus):
		print('Epoch : {0}/{1} ,iter: {2}/{3} ({4}%)'.format(i+ 1, no_epochs, int(start_iter/sequence_len), 
			total_iter, int((start_iter/float(sequence_len))*100/float(total_iter))))
		
		train_chunk = text_corpus[start_iter : start_iter + sequence_len]
		target_chunk = text_corpus[start_iter + 1 : start_iter + sequence_len + 1]

		train = make_chars_input(train_chunk, vocab)
		target = make_chars_input(target_chunk, vocab)
		
		hidden = network.init_hidden()
		network.zero_grad()
		loss = 0
		for c in range(train.size()[0]):
			output, hidden = network(train[c], hidden)
			loss += criterion(output,target[c])

		loss.backward()
		optimizer.step()
		iter_loss.append(loss.data[0]/sequence_len)
		avg_loss += loss.data[0]/sequence_len
		print('Loss: {0}'.format(loss.data[0]/sequence_len))

		start_iter = start_iter + sequence_len

	avg_loss = avg_loss / float(total_iter)
	epoch_loss.append(avg_loss)

	#if i in epochs_evalute:
		# TO DO : evaluate