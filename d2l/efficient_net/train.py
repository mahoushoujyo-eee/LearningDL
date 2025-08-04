import torch, torch.nn as nn, torch.optim as optim
from efficientnet import efficientnet_b0
from data import get_loaders
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = efficientnet_b0(num_classes=10).to(device)
train_loader, test_loader = get_loaders()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean()

for epoch in range(50):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 验证
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total   += y.size(0)
    print(f'Epoch {epoch}: Acc = {correct/total:.4f}')