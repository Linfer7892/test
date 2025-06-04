import argparse
import os

import models
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def test(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (inputs, labels) in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

def main(args):
    # 선택한 모델과 dataset 로딩
    if args.dataset.lower() == "cifar10" :
        num_classes = 10
    else :
        num_classes = 100
    model = models.load_model(args.model, num_classes=num_classes)

    train_dataset = datasets.load_dataset(args.dataset, train=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False)
    
    # DataLoader 생성: (여기서는 PyTorch의 DataLoader를 사용)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 옵티마이저 정의 (예: SGD)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 간단한 training loop (에포크 수만큼 반복)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(args.num_epochs):
        model.train()

        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        
        val_loss, val_acc = test(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}%")

    print("Training finished.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet34", help="Choose the model: resnet34, densenet, fractalnet")
    argparser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use: cifar10 or cifar100")
    argparser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    argparser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = argparser.parse_args()
    main(args)
