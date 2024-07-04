# A Simple RNN Task: 利用RNN的二元分类网络区分不同函数
# cmd> pip install torch numpy matplotlib 

#导入必要的库：PyTorch用于神经网络的构建和训练，NumPy用于数据处理，Matplotlib用于数据可视化。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
DATA_SIZE是数据总量，设为1000。
sine_data_size和sigmoid_data_size分别是正弦函数和S形函数数据的大小，通过随机选择使总数为1000。
steps定义了从0到10的步长为0.5的数组。
随机生成参数a和b用于正弦函数和S形函数，并根据这些参数生成数据。
"""
DATA_SIZE = 1000

sine_data_size = np.random.randint(int(0.3 * DATA_SIZE), int(0.7 * DATA_SIZE))
sigmoid_data_size = DATA_SIZE - sine_data_size

steps = np.arange(0, 10, 0.5)

# generate sine-like function samples
sine_init = np.random.uniform(-3, 3, (sine_data_size, 2))  # randomize a and b for sin(ax+b)
sine_data = np.sin(sine_init[:, :1] * steps + sine_init[:, 1:])

# generate sigmoid-like function samples
sigmoid_init = np.random.uniform(-3, 3, (sigmoid_data_size, 2)) # randomize a and b for 1/(1+e^(-ax+b))
sigmoid_data = 1 / (1 + np.exp(0 - sigmoid_init[:, :1] * steps + sigmoid_init[:, 1:]))

#可视化生成的正弦和S形函数数据。
fig, axs = plt.subplots(1, 2)
axs[0].plot(sine_data[0])
axs[1].plot(sigmoid_data[1])
plt.show()

#为每个样本添加标签：正弦函数数据标记为1，S形函数数据标记为0。将数据转换为PyTorch张量。
# mix data
sine_data = np.concatenate((sine_data, np.ones((sine_data_size, 1))), axis=1)
sigmoid_data = np.concatenate((sigmoid_data, np.zeros((sigmoid_data_size, 1))), axis=1)
data = np.concatenate((sine_data, sigmoid_data), axis=0)
data = torch.Tensor(data)

#将数据集按80:20划分为训练集和测试集。
# split two datasets
from torch.utils.data import random_split
train_set, test_set = random_split(data, [0.8, 0.2])

"""
SimpleClassificationRNN类定义了一个简单的RNN模型，包括一个RNN层和一个线性层。
forward方法定义了前向传播过程。
"""
# define network
class SimpleClassificationRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleClassificationRNN, self).__init__()
        '''
        task 1: write network structure here using nn.RNN
        '''
        self.rnn = nn.RNN(input_size=1,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=1)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, seq, hc=None):
        '''
        task 2: write forward process
        '''
        tmp, hc = self.rnn(seq, hc)
        out = torch.sigmoid(self.linear(hc[-1, ..., :]))
        return out, hc

#设置RNN隐藏层大小为16，学习率为0.01。使用二元交叉熵损失函数（BCELoss）和Adam优化器。
hidden_size = 16
learning_rate = 0.01

model = SimpleClassificationRNN(hidden_size)

'''
task 3: select appropriate criterion and optimizer
'''
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

#定义计算准确率的函数，将预测值大于0.5的视为1，小于等于0.5的视为0。
def cal_accuracy(preds, true_values):
    preds = torch.where(preds>0.5, 1, 0)
    acc = torch.sum(1-torch.abs(preds-true_values)) / preds.shape[0]
    return acc

#设置训练轮数为500。在每个训练轮次中，计算损失和准确率，并更新模型参数。每10轮打印一次损失和准确率。
# training ...
epochs = 500
loss_log = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output, _ = model(train_set[:][:, :-1, np.newaxis])
    loss = criterion(output.view(-1), train_set[:][:, -1])
    acc = cal_accuracy(output.view(-1), train_set[:][:, -1])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch {}: loss {} acc {}".format(epoch, loss.item(), acc))
# performance on test set

#在测试集上计算损失和准确率，并打印结果。
output, _ = model(test_set[:][:, :-1, np.newaxis])
loss = criterion(output.view(-1), test_set[:][:, -1])
acc = cal_accuracy(output.view(-1), test_set[:][:, -1])

print("Test set: loss {} acc {}".format(loss.item(), acc))