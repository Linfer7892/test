import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

import models
import datasets

def train(model, dataloader, optimizer, criterion, device):
    model.train()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(model, dataloader, criterion, device):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.load_model(args.model).to(device)

    train_dataset = datasets.load_dataset(args.dataset, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = datasets.load_dataset(args.dataset, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        train(model, train_loader, optimizer, criterion, device)
        test(model, test_loader, criterion, device)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--dataset", type=str, default="cifar10")
    argparser.add_argument("--num_epochs", type=int, default=100)
    argparser.add_argument("--batch_size", type=int, default=64)
    args = argparser.parse_args()
    main(args)
