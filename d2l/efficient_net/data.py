import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_loaders(root='./data', batch_size=128, num_workers=2):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]
    train_tf = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    test_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    train_ds = CIFAR10(root, train=True,  download=True, transform=train_tf)
    test_ds  = CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, test_loader