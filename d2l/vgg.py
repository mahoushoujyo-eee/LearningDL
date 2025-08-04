import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

# 修改后的VGG模型定义
def vgg(conv_arch, input_size=28):
    conv_blks = []
    in_channels = 1  # 输入通道数，FashionMNIST是灰度图
    
    # 计算经过池化后的特征图尺寸
    current_size = input_size
    for (num_convs, _) in conv_arch:
        # 每次池化尺寸减半
        current_size = current_size // 2
        if current_size < 1:
            raise ValueError(f"特征图尺寸在{len(conv_blks)+1}个卷积块后变为0")
    
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        blk = []
        for _ in range(num_convs):
            blk.append(nn.Conv2d(in_channels, out_channels, 
                                kernel_size=3, padding=1))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv_blks.append(nn.Sequential(*blk))
    
    # 计算全连接层输入尺寸
    fc_input_size = out_channels * current_size * current_size
    
    return nn.Sequential(
        *conv_blks, 
        nn.Flatten(),
        # 精简全连接层部分
        nn.Linear(fc_input_size, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 10))



def main():
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 原始VGG架构参数 - 减少池化层数量
    conv_arch = ((1, 64), (1, 128), (2, 256))  # 只使用3个卷积块
    
    # 缩小模型规模
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    
    # 训练参数
    lr, num_epochs, batch_size = 0.005, 10, 128  # 降低学习率，增加epoch

    # 数据预处理 - 使用FashionMNIST专用标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # FashionMNIST专用标准化
    ])

      # 数据预处理 - 添加数据增强
    # 训练集使用增强变换，测试集使用基本变换
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        transforms.RandomRotation(15),            # 随机旋转±15度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # FashionMNIST专用标准化
    ])

    # 创建数据加载器
    train_iter = DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                             transform=train_transform),
        batch_size=batch_size, shuffle=True)
    
    test_iter = DataLoader(
        datasets.FashionMNIST('../data', train=False, download=True,
                             transform=test_transform),
        batch_size=batch_size, shuffle=False)
    
    try:
        # 初始化模型
        net = vgg(small_conv_arch, input_size=28).to(device)
        print("模型架构:")
        print(net)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in net.parameters())
        print(f"模型总参数: {total_params:,}")
    except ValueError as e:
        print(f"模型初始化错误: {e}")
        # 尝试使用更少的卷积层
        conv_arch = conv_arch[:2]  # 只使用2个卷积块
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        net = vgg(small_conv_arch, input_size=28).to(device)
        print("使用简化模型架构")
        print(net)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_iter):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每100个batch打印一次
            if batch_idx % 100 == 0:
                acc = 100. * correct / total
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_iter)} '
                      f'| Loss: {loss.item():.4f} | Acc: {acc:.2f}%')
        
        # 测试阶段
        net.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # 计算准确率
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), 'best_fashion_mnist_vgg.pth')
        
        # 打印epoch总结
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss/len(train_iter):.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss/len(test_iter):.4f} | Test Acc: {test_acc:.2f}%')
        print(f'  Best Test Acc: {best_acc:.2f}%\n')
    
    print(f"训练完成! 最佳测试准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()

