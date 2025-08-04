import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Inception Module ---
# No major issues here, but the super().__init__() call can be simplified.
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        # CORRECTED: Simplified the super() call for modern Python.
        # The original `**kwargs` was not used, so it's removed.
        super().__init__()
        
        # Branch 1: 1x1 convolution
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # Branch 2: 1x1 followed by 3x3 convolution
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # Branch 3: 1x1 followed by 5x5 convolution
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        # Concatenate the outputs of the four branches along the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)

# --- Weight Initialization Function (Recommended Addition) ---
def init_weights(m):
    """Initializes weights for Conv2d and Linear layers for better training."""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def main():
    # --- Model Architecture (GoogLeNet Blocks) ---
    # The architecture itself is logically sound.
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    
    # --- Hyperparameters ---
    lr, num_epochs, batch_size = 0.001, 10, 128

    # --- Data Preprocessing ---
    # IMPROVEMENT: Resizing to 96x96 is a common practice for adapting smaller
    # datasets to GoogLeNet-like architectures to prevent feature maps from
    # becoming too small too quickly. 64x64 also works.
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- Data Loaders ---
    train_iter = DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(
        datasets.FashionMNIST('../data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False)

    # --- Device Configuration and Model Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    net.to(device)
    # IMPROVEMENT: Apply the custom weight initialization.
    net.apply(init_weights)
    
    print("Model architecture initialized.")
    # print(net) # You can uncomment this to see the full architecture

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    # --- Learning Rate Scheduler ---
    # CRITICAL FIX: Changed mode from 'max' to 'min'.
    # We want to reduce the learning rate when the validation *loss* stops
    # *decreasing*, so we monitor a value that should be minimized.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # --- Training and Validation Loop ---
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_iter):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | '
                      f'Loss: {train_loss / (batch_idx + 1):.3f} | '
                      f'Acc: {100. * correct / total:.3f}% ({correct}/{total})')
        
        # CRITICAL FIX: Calculate the average training loss for the epoch summary.
        # This variable was used in the final print statement but never defined.
        avg_train_loss = train_loss / len(train_iter)

        # --- Validation Phase ---
        net.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                # Accumulate total loss for the test set
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        avg_test_loss = test_loss / test_total
        test_acc = 100. * test_correct / test_total
        
        # Update the learning rate scheduler based on validation loss
        scheduler.step(avg_test_loss)
    
        # Print summary for the completed epoch
        print(f'\n--- End of Epoch {epoch} ---')
        print(f'Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%\n')

if __name__ == '__main__':
    main()
