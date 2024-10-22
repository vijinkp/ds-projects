# https://cs224d.stanford.edu/notebooks/vanishing_grad_example.html
# https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle

class Net(nn.Module):
    def __init__(self, init_weights = True):
        super(Net, self).__init__()
        self.init_weights = init_weights
        self.dropout_prob = 0.25
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        #self.conv2_drop = nn.Dropout2d(p=self.dropout_prob)
        self.fc1 = nn.Linear(320, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.bn4 = nn.BatchNorm1d(10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.uniform_(m.weight)
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.sigmoid(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.sigmoid(self.fc1(x))
        x = F.relu(self.bn3(self.fc1(x)))
        #x = F.dropout(x, training=self.training, p=self.dropout_prob)
        #x = self.fc2(x)
        x = self.bn4(self.fc2(x))
        return F.log_softmax(x, dim=1)

def train(args, model, use_cuda, train_loader, optimizer, epoch):
    criterion = nn.NLLLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test(args, model, use_cuda, test_loader):
    criterion = nn.NLLLoss(size_average=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data.item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss, 100. * correct / float(len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/vparambath/Desktop/iith/ds-projects/data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/vparambath/Desktop/iith/ds-projects/data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    if use_cuda:
        model = model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_loss_map = {}
    train_accuracy_map = {}
    test_loss_map = {}
    test_accuracy_map = {}
    for epoch in range(1, args.epochs + 1):
        train(args, model, use_cuda, train_loader, optimizer, epoch)
        train_loss, train_accuracy = test(args, model, use_cuda, train_loader)
        test_loss, test_accuracy = test(args, model, use_cuda, test_loader)
        train_loss_map[epoch] = train_loss
        train_accuracy_map[epoch] = train_accuracy
        test_loss_map[epoch] = test_loss
        test_accuracy_map[epoch] = test_accuracy

    with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_train_loss_map.pkl', 'wb') as fp:
        pickle.dump(train_loss_map, fp)

    with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_train_accuracy_map.pkl', 'wb') as fp:
        pickle.dump(train_accuracy_map, fp)

    with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_test_loss_map.pkl', 'wb') as fp:
        pickle.dump(test_loss_map, fp)

    with open('/home/vparambath/Desktop/iith/ds-projects/data/mnist/relu_20_bn_woDP_normal_test_accuracy_map.pkl', 'wb') as fp:
        pickle.dump(test_accuracy_map, fp)


if __name__ == '__main__':
    main()