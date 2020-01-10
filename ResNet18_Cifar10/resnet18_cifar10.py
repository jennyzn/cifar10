#!/usr/bin/env python3
# coding=utf-8

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import matplotlib.pyplot as plt
import cv2 as cv

# 超参 
batch_size = 256
num_epochs = 90
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
CIFAR10:
10类6w图,每类6000,5000训练1000测试
3x32x32彩图
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""
# step1: 数据预处理与加载数据
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='~/file/datasets/pytorch', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='~/file/datasets/pytorch', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8) # num_workers=4
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8)

# step2: 构造网络
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# step3:损失函数优化方法
model = ResNet(ResidualBlock).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)# 没有to方法 .to(device)


# step4: 训练与评估
def eval():
    model.eval()
    eval_loss, eval_acc =0., 0.
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        eval_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        eval_acc += (pred == labels).float().mean()
    return eval_loss/len(test_loader), eval_acc/len(test_loader)


def train(start_epoch=0):
    model.train()
    for epoch in range(start_epoch, num_epochs):
        print('*' * 40), print(f'epoch {epoch}')
        train_loss, train_acc = 0., 0.
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            train_acc += (pred == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f'[epoch:{epoch}, step:{step}] loss:{train_loss/step:.4f}, acc:{train_acc/step:.4f}')

        # 评估
        eval_loss, eval_acc = eval()
        print(f'[eval: loss:{eval_loss:.4f}, acc:{eval_acc:.4f}]')
        # 保存模型, 仅网络中的参数
        print(f'Saving epoch {epoch} model ...')
        checkpoint = {'start_epoch':epoch+1, 'state_dict':model.state_dict()}
        torch.save(checkpoint, 'checkpoint/ResNet18_cifar10_epoch%d.ckpt'%(epoch))

    print('Finished Training')
    # 保存最终模型
    checkpoint = {'state_dict':model.state_dict()}
    torch.save(checkpoint, 'ResNet18_cifar10.ckpt')
    # 打包成手机可运行的模型, 放到cuda里跑的模型不能直接
    # save2mobile(model)



"""
        Utils
"""

def save2mobile(model):
    model.eval()
    example = torch.rand(1,3,32,32)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("ResNet.pt")

def get_untrain_model():
    model = ResNet(ResidualBlock)
    save2mobile(model)

# 从保存的模型中加载
def load_module(model_file):
    model = ResNet(ResidualBlock) # 不加这句会使用全局变量model，而且不能加cuda，否则输入也得加.cuda且转成手机模型会挂
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    return model

# 从保存的模型中加载指定epoch的参数,继续训练
def continue_train(model_file):
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['start_epoch']
    train(start_epoch)

# 查看输入的shape并且从数据集中搞出一张图片
def get_test_img():
    img = train_loader.dataset.data[0]
    # img = train_loader.dataset.train_data[2]
    print("数据图片尺寸: ", img.shape) #32x32x3 
    plt.imshow(img)
    plt.show()
    plt.imsave("./Cifar10_test.jpg", img)

    for inputs, labels in test_loader:
        print("一批次输入尺寸: ", inputs.shape)
        break

# 以数据集图片进行预测
def demo_test(model):
    # 图片预处理
    img = cv.imread("Cifar10_test.jpg")
    cv.imshow("test", img)
    cv.waitKey(0)
    print(type(img))
    # ???缩放一下
    img = cv.resize(img,(32,32))
    img = torch.from_numpy(img)
    img = img.reshape(1,3,32,32).float()
    # print(type(img))
    out = model(img)
    _, pred = torch.max(out, 1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"预测类别: {pred.data} 含义: {classes[pred.data]}")


if __name__ == '__main__':
    # get_untrain_model() # 测试一下直接将网络转成参数文件，安卓端导入是否成功
    # get_test_img() # 从数据集中拿出一张图片
    # train() # 正常训练，可以随时<C-c>终止
    # continue_train("./checkpoint/ResNet18_cifar10_epoch69.ckpt") # 继续训练，指定继续的epoch文件

    # model = load_module("./checkpoint/ResNet18_cifar10_epoch66.ckpt")
    # demo_test(model) # 测试数据集中拿出的图片，网络是否预测对

    model = load_module("./checkpoint/ResNet18_cifar10_epoch66.ckpt")
    save2mobile(model)


# 准确率
# [eval: loss:0.3399, acc:0.9113]


