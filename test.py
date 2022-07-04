import torch
import torch.nn as nn

inputs = torch.FloatTensor(1, 1, 28, 28)

in_channels, out_channels, kernel_size = 1, 32, 3

conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=1, padding=1)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
# torch.max_pool2d() 랑 nn.MaxPool2d() 차이?
out1 = conv1(inputs)
print(out1.shape)
out2 = pool1(out1)
print(out2.shape)
