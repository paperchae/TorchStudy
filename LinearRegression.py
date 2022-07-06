import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(1)


x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
# method 1
# W = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
# method 2
# model = nn.Linear(1, 1)  # input_dim(x)=1, output_dim(y)=1
# print(list(model.parameters()))
# method 3
class LinearRegressionModel(nn.Module):
    # __init__에서 모델의 구조와 동작을 정의하는 생성자를 정의
    # 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으로 호출
    def __init__(self):
        # nn.Module 클래스의 속성들을 가지고 초기화
        super().__init__()
        self.linear = nn.Linear(1, 1)

    # 모델이 학습 데이터를 입력받아서 forward 연산을 진행하는 함수
    def forward(self, inputs):
        return self.linear(inputs)


model = LinearRegressionModel()

cost_record = []
# optimizer 설정
# optimizer = optim.SGD([W, b], lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.001)

nb_epochs = 1999

for epoch in range(nb_epochs + 1):
    # hypothesis 설정
    # hypothesis = W * x_train + b
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    # cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)
    cost_record.append(cost.item())

    # pytorch는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적하여 합하는 특징이 있으므로 미분값을 계속 0으로 초기화 필요
    optimizer.zero_grad()
    # cost 함수를 미분하여 gradient 계산
    cost.backward()
    # W, b 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
e = np.arange(0, nb_epochs + 1)
print(len(e), e)
print(len(cost_record), cost_record)

plt.plot(e, cost_record)
plt.show()

new_x = torch.FloatTensor([4])
pred_y = model(new_x)
print('model parameters : ', list(model.parameters()))
print('ground truth = 8, pred_y =', pred_y)
