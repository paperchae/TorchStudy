import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# TensorDataset, DataLoader 를 사용하면 미니배치학습, 데이터 셔플, 병렬 처리까지 간단히 수행 가능
# Dataset을 정의하고, 이를 DataLoader에 전달
torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152],
                             [185],
                             [180],
                             [196],
                             [142]])

# Tensor를 입력받아 이를 데이터셋으로 변환하는 TensorDataset
dataset = TensorDataset(x_train, y_train)

# DataLoader는 기본적으로 2개 인자를 받음 (dataset, mini batch size <- 2의 배수 사용)
# shuffle=True <- Epoch마다 데이터셋 섞음
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


class MultiVariableLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, inputs):
        return self.linear(inputs)


model = MultiVariableLinearRegression()

nb_epochs = 2000

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(nb_epochs):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx+1)
        print(samples)

        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))

new_x = torch.FloatTensor([73, 80, 75])

pred_y = model(new_x)
print(pred_y)