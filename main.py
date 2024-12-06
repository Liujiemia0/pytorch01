import torch
import torch.nn as nn
import pandas as pd
from sympy import resultant

raw_df = pd.read_csv( 'train.csv')
#特征
#标签
label = raw_df['label'].values

raw_df = raw_df.drop(['label'],axis=1)
feature = raw_df.values
#整个数据集划分为两个数据集，训练集，测试集
train_feature = feature[:int(len(feature)*0.8)]
train_label = label[:int(len(label)*0.8)]
test_feature = feature[int(len(feature)*0.8):]
test_label = label[int(len(label)*0.8):]
# numpy数据格式转化为tensor格式
train_feature = torch.tensor(train_feature).to(torch.float).cuda()
train_label = torch.tensor(train_label).cuda()
test_feature = torch.tensor(test_feature).to(torch.float).cuda()
test_label = torch.tensor(test_label).cuda()
#将神经网络视作一个黑盒子函数
#784个像素点构成的灰度图-->函数————>10个概率（0-9的概率）

#定义网络结构
#输入层
#隐藏层
#隐藏层
#输出层

#验证模型是否合理
data = torch.rand(1,784)
model = nn.Sequential(
    nn.Linear(784,444),
    nn.ReLU(),
    nn.Linear(444,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,10),
    # nn.Softmax(),
)
model = model.cuda()
#梯度下降（找到一组合适的w和b让损失值）越小越好
#瞎子下山
# 分类问题用交叉熵损失函数
lossfunction = nn.CrossEntropyLoss()
#优化器Adam
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)
#训练轮数
for i in range(10):
    # 清空优化器的梯度（偏导数），Adam中会存放之前训练的梯度
    optimizer.zero_grad()
    predict = model(train_feature)
    #打印学习率
    result = torch.argmax(predict,dim = 1)
    train_acc = torch.mean((result==train_label).to(torch.float))
    loss = lossfunction(predict,train_label)
    # 反向传播
    loss.backward()
    # 梯度下降
    optimizer.step()
    #打印损失函数
    print('train loss:{} train acc:{}'.format(loss.item(),train_acc.item()))

# #保存模型
torch.save(model.state_dict(),'./mymodel.pt')

#加载模型文件里的参数（w、b）
params = torch.load('./mymodel.pt')
#把参数塞进模型里
model.load_state_dict(params)

new_test_data = test_feature[100:111]
new_test_label = test_label[100:111]
predict = model(new_test_data)
result = torch.argmax(predict,dim=1)
print(new_test_label)
print(result)