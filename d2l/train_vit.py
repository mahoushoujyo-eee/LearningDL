from vit import ViT
from torchvision import datasets, transforms
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

model = ViT(
    image_size=32,
    patch_size=4,       # 或者patch_size=2
    num_classes=10,     # CIFAR-10 或 100 for CIFAR-100
    dim=512,
    depth=6,
    heads=12,
    mlp_dim=1024,
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(testloader)
    acc = 100. * correct / total
    return test_loss, acc

def main():
    model.apply(init_weights)
    model.to(device)

    # 训练模型
    num_epochs = 10
    print("开始训练...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s')
    
    print("训练完成!")


if __name__ == '__main__':
    main()