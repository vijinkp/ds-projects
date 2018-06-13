# https://github.com/spro/char-rnn.pytorch
# https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
# https://github.com/fastai/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb
import glob, unicodedata, string
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

torch.manual_seed(777)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="lstm", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
       	return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

def char_tensor(string, vocab):
    tensor = np.zeros(len(string))
    for c in range(len(string)):
        tensor[c] = vocab.index(string[c])
    return torch.Tensor(tensor).view(1,-1)

def one_hot(x, num_classes):
	idx = x.long()
	idx = idx.view(-1, 1)
	x_one_hot = torch.zeros(x.size()[0] * x.size()[1], num_classes)
	x_one_hot.scatter_(1, idx, 1)
	x_one_hot = x_one_hot.view(x.size()[0], x.size()[1], num_classes)
	return x_one_hot

# corpus creation
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in vocab)

def generate(network, vocab, prime_str='A', predict_len=200, temperature=0.8):
    hidden = network.init_hidden(1)
    prime_input = Variable(one_hot(char_tensor(prime_str, vocab).unsqueeze(0), len(vocab)))
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = network(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
    	output, hidden = network(inp, hidden)
    	output_dist = output.data.view(-1).div(temperature).exp()
    	top_i = torch.multinomial(output_dist, 1)[0]
    	predicted_char = vocab[top_i]
    	predicted += predicted_char
    	inp = Variable(one_hot(char_tensor(predicted_char, vocab).unsqueeze(0), len(vocab)))
    return predicted

# Extended ascii of 256 characters
vocab = ''.join([chr(char) for char in range(256)])
root_data_folder = '/home/vparambath/Desktop/iith/ds-projects/data/rnn/cleaned_text_data'
sequence_len = 200
no_epochs = 100
epochs_evaluate = [20, 40, 60, 80, 100]
hidden_size = 100
lr = 0.005


text_corpus = []
for file in glob.glob('{0}/*.txt'.format(root_data_folder)):
	with open(file, 'r') as fp:
		text_corpus.append(unicode_to_ascii(fp.read()))
text_corpus = ' '.join(text_corpus)

# initilaise network
network = CharRNN(len(vocab), hidden_size, len(vocab))
print(network)

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# train network
total_iter = int(len(text_corpus)/sequence_len)
epoch_loss = []
gen_text = []
for i in range(no_epochs):
	iter_loss = []
	avg_loss = 0
	start_iter = 0
	optimizer.zero_grad()
	while start_iter < len(text_corpus):
		print('Epoch : {0}/{1} ,iter: {2}/{3} ({4}%)'.format(i+ 1, no_epochs, int(start_iter/sequence_len), 
			total_iter, int((start_iter/float(sequence_len))*100/float(total_iter))))
		
		train_chunk = text_corpus[start_iter : start_iter + sequence_len]
		target_chunk = text_corpus[start_iter + 1 : start_iter + sequence_len + 1]

		if not len(train_chunk) == sequence_len:
			break

		train = Variable(one_hot(char_tensor(train_chunk, vocab), len(vocab)))
		target = Variable(char_tensor(target_chunk, vocab).type(torch.LongTensor))

		hidden = network.init_hidden(1)
		network.zero_grad()
		loss = 0
		for c in range(sequence_len):
			output, hidden = network(train[:,c], hidden)
			loss += criterion(output.view(1, -1), target[:,c])

		loss.backward()
		optimizer.step()

		iter_loss.append(loss.data.item()/sequence_len)
		avg_loss += loss.data.item()/sequence_len
		print('Loss: {0}'.format(loss.data.item()/sequence_len))
		start_iter = start_iter + sequence_len
		
	avg_loss = avg_loss / float(total_iter)
	epoch_loss.append(avg_loss)

	if i+1 in epochs_evaluate:
		gen_text.append(generate(network, vocab))

# writing results
with open('lstm_epoch_loss.pkl', 'wb') as fp:
	pickle.dump(epoch_loss, fp)

with open('lstm_gen_text.pkl', 'wb') as fp:
	pickle.dump(gen_text, fp)

