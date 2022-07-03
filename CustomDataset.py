import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)


class CustomDataset(Dataset):
    # dataset의 전처리를 해주는 부분
    def __init__(self):
        self.x_train = torch.FloatTensor([[73, 80, 75],
                                          [93, 88, 93],
                                          [89, 91, 80],
                                          [96, 98, 100],
                                          [73, 66, 70]])
        self.y_train = torch.FloatTensor([[152],
                                          [185],
                                          [180],
                                          [196],
                                          [142]])

    # dataset의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return len(self.x_train)

    # dataset에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor(self.y_train[idx])
        return x, y


class MultiLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, inputs):
        return self.linear(inputs)


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MultiLinearRegression()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000

for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
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

