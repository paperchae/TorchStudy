import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_dataset/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_dataset/', train=False, transform=transforms.ToTensor(), download=True)

dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


# 1번 레이어 : Conv1(in_channel=1, out_channel=32,
#                   kernel_size=3, stride=1,padding=1) + ReLu + MaxPool(2)

# 2번 레이어 : Conv2(in_channel=32, out_channel=64,
#                   kernel_size=3, stride=1, padding=1) + ReLu + MaxPool(2)

# 3번 레이어 : FC Layer
# batch_size * 7 * 7 * 64 -> batch_size * 3136
# 전결합층 (뉴런 10개) + softmax


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                               kernel_size=kernel_size, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs):
        conv1 = torch.relu(self.conv1(inputs))
        print('conv1 + relu :', conv1.shape)
        pool1 = self.pool1(conv1)
        print('pool1 :', pool1.shape)
        conv2 = torch.relu(self.conv2(pool1))
        print('conv2 + relu :', conv2.shape)
        pool2 = self.pool2(conv2)
        print('pool2 :', pool2.shape)
        out = pool2.view(pool2.size(0), -1)
        out = self.fc(out)
        print('out :', out.shape)
        return out


# class Linear(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#
#     def forward(self, inpuuts):
#         out = self.linear(inpuuts)
#         print(out.shape)
#         return out


# cnn_input = torch.FloatTensor(1, 1, 28, 28)
cnn_model = CNN(1, 32, 3).to(device)
#
# linear_model = Linear(3136, 10).to(device)
# linear_out = linear_model(cnn_out)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

total_batch = len(dataloader)
print('batchN :', total_batch)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        prediction = cnn_model(X)

        optimizer.zero_grad()
        cost = criterion(prediction, Y)
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

#
# #
# # Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
# #                   stride=1, padding=1)
# # Conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size,
# #                   stride=1, padding=1)
# # pool = nn.MaxPool2d(kernel_size=2, stride=2)
# # # print(Conv1)
# # out1 = Conv1(inputs)
# # print('conv1 :', out1.shape)
# # out2 = pool(out1)
# # print('pool1 :', out2.shape)
# # out3 = Conv2(out2)
# # print('conv2 :', out3.shape)
# # out4 = pool(out3)
# # print('pool2 :', out4.shape)