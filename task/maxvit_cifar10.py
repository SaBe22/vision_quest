import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2

from model.maxvit import MaxVit

NORMALIZE_MEAN: list[float] = [0.5, 0.5, 0.5]
NORMALIZE_STD: list[float] = [0.5, 0.5, 0.5]

BATCH_SIZE: int = 512

def train_epoch(
    model, train_loader, criterion, optimizer, scheduler=None, device="cpu"
):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # Move the batch to the device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Calculate the loss
        loss = criterion(model(images), labels)
        # Backpropagate the loss
        loss.backward()
        # Update the model's parameters
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(images)

    return total_loss / len(train_loader.dataset)


def evaluate(model, test_loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move the batch to the device
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            logits = model(images)

            # Calculate the loss
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(images)

            # Calculate the accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()

    accuracy = correct / len(test_loader.dataset)
    test_loss = total_loss / len(test_loader.dataset)

    return accuracy, test_loss


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
):
    """
    Train the model for the specified number of epochs.
    """
    # Keep track of the losses and accuracies
    train_losses, test_losses, accuracies = [], [], []
    # Train the model
    for i in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler=scheduler,
            device=device,
        )
        accuracy, test_loss = evaluate(model, test_loader, criterion, device=device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        print(
            f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    return train_losses, test_losses, accuracies


if __name__ == "__main__":
    img_size = (64, 64)
    output_dim = 10
    embedding_dims = [32, 64, 128, 256]
    block_depths = [2, 2, 4]
    num_heads = [32, 32, 32]
    window_size = (4, 4)
    dropout = 0.
    stochastic_dropout = 0.1

    train_transform = v2.Compose([
        v2.Resize(size=img_size, antialias=True),
        v2.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    test_transform = v2.Compose([
        v2.Resize(size=img_size, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="data/cifar-10", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="data/cifar-10", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaxVit(
        output_dim=10,
        embedding_dims=embedding_dims,
        window_size=window_size,
        block_depths=block_depths,
        num_heads=num_heads,
        dropout=dropout,
        stochastic_dropout=stochastic_dropout
    )

    model.to(device)
    epochs = 50
    lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, epochs, loss_fn, optimizer, scheduler=scheduler, device=device)