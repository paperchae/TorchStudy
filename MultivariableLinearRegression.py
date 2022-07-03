import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# 모델 초기화
# W = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
# model = nn.Linear(3, 1)
class MultivariableLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.multilinear = nn.Linear(3, 1)

    def forward(self, inputs):
        return self.multilinear(inputs)


model = MultivariableLinearRegression()
# optimizer 설정
# optimizer = optim.SGD([W, b], lr=1e-5)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    # hypothesis = x_train.matmul(W) + b
    prediction = model(x_train)

    # cost 계산
    # cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# model test
test_x = torch.FloatTensor([[73, 80, 75]])
pred_y = model(test_x)
print('model parameters : ', list(model.parameters()))
print('ground truth = 152, pred_y =', pred_y)
