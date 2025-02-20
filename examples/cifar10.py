from efficient_kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

gridupdate_samples = DataLoader(trainset, batch_size=4000, shuffle=False, )

# Define the KAN model
zero_func = lambda x: x*0
model = KAN([3*32*32, 3, 3, 3, 10], grid_eps=1, base_activation=zero_func, grid_size=5, spline_order=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
print(f"Model has {model.num_params()} number of parameters.")

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

# Define loss
criterion = nn.CrossEntropyLoss()
grid_update_freq=1
layer_num = 1
for epoch in range(40):
    # Train
    if (epoch % grid_update_freq == 0 and epoch != 0):
        images, labels = next(iter(gridupdate_samples))
        images = images.view(-1, 3*32*32).to(device)
        print(images.shape)
        model.update_layer_grid(images, margin=0.1, layer=layer_num)
        layer_num = 1 if layer_num == 3 else layer_num + 1
        # optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        print("Grid updated!")

    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 3*32*32).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            reg = 1e-6*model.regularization_loss()
            loss_reg = loss + reg
            # loss_reg.backward()
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=f'{loss.item():.6f}',reg=f'{reg.item():.4f}', accuracy=f'{accuracy.item():.4f}', lr=f"{optimizer.param_groups[0]['lr']:.5f}")

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 3*32*32).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )


