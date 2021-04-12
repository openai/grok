from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle
import os
import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == len(train_loader) - 1 or batch_idx % epoch == 0:
            with torch.no_grad():
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                accuracy = pred.eq(target.view_as(pred)).to(torch.float32).mean()
            logger.log({'train_loss': loss.item(),
                        'train_accuracy': accuracy.item(),
                        'epoch': epoch - 1 + batch_idx / len(train_loader),
                        'step': (epoch - 1) * len(train_loader) + batch_idx})
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, logger, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.log({'valid_loss': test_loss,
                'valid_accuracy': correct/len(test_loader.dataset),
                'epoch': epoch})

class Logger:
    def __init__(self, filename):
        self.logs = []
        self.filename = filename
    
    def log(self, data):
        self.logs.append(data)

    def flush(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.logs, f)
            
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data_pct', type=int, default=100)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(-1)
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    dataset1.data = dataset1.data[:int(len(dataset1) * args.data_pct / 100)]
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logger = Logger(os.path.join(os.environ['RCALL_LOGDIR'], f'log_{now}.pkl'))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, logger)
        test(model, device, test_loader, logger, epoch)
        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.flush()
            

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()