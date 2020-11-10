import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# 1. Basic autograd example 1
# 创建张量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 建立方程
y = w * x + b

# 计算梯度
y.backward()

# 打印梯度
print(x.grad)
print(w.grad)
print(b.grad)


# 2. Basic autograd example 2
# 创建张量，随机，维度：(10,3) (10,2)
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 建立一个全链接层
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# 建立损失函数和 optimizer
criterion = nn.MSELoss()    # 均方差
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)   # 随机梯度下降

# 前向传播
pred = linear(x)
print(pred)

# 计算损失值
loss = criterion(pred, y)
print('loss: ', loss.item())

# 反向传播
loss.backward()

# 打印梯度
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 单步梯度下降
optimizer.step()

pred = linear(x)
loss = criterion(pred, y)

print('loss after 1 step optimization:', loss.item())


# 3. Loading data from numpy
# 建立 numpy array
x = np.array([[1, 2], [3, 4]])

# numpy array --> torch tensor
y = torch.from_numpy(x)

# torch tensor --> numpy array
z = y.numpy()

print(x)
print(y)
print(z)


# 4. Input pipeline
# 下载并构建数据集
train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# 获取数据
image, label = train_dataset[0]
print(image.size())
print(label)

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# 迭代，从文件加载数据
data_iter = iter(train_loader)

# mini-batch images and labels
images, labels = data_iter.next()

# 实际使用数据加载器的方法
for images, label in train_loader:
    # <TODO>
    # <训练代码>
    pass


# 5. Input pipeline for custom dataset
# 建立自己的数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # <TODO>
        # 1. 初始化文件路径或文件名列表
        pass

    def __getitem__(self, index):
        # <TODO>
        # 1. 从文件读一个数据（例如：numpy.fromfile）
        # 2. 处理数据（例如：torchvision.Transform.）
        # 3. 返回数据对（例如：image and label）
        pass

    def __len__(self):
        # 返回数据集的大小
        return 0


# 使用预先建立的 data loader
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


# 6. Pretrained model
# 下载并加载预训练模型 ResNet-18
resnet = torchvision.models.resnet18(pretrained=True)

# 如果只想微调模型的顶层，请设置如下:
for param in resnet.parameters():
    param.requires_grad = False

# 更换顶层进行微调
resnet.fc = nn.Linear(resnet.fc.in_features, 100)   # 举例 100

# 前向传播
images = torch.randn(63, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())   # (64, 100)


# 7. Save and load the model
# Save and load the entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters(recommended)
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
