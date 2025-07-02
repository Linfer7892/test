import argparse
import os

import models
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def main(args):
    # 모델 로드
    model = models.load_model(args.model, args.dataset)
    
    # 데이터셋 로드
    train_dataset = datasets.load_dataset(args.dataset, train=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # optimizer, criterion, scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # 결과를 저장할 리스트 초기화
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        current_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # print_freq마다 출력
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {current_train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*correct_train/total_train:.2f}%')
        
        # 에포크 종료 후 학습 결과 저장
        train_losses.append(current_train_loss / len(train_loader))
        train_accuracies.append(100. * correct_train / total_train)

        # eval_freq마다 평가
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            current_test_loss = 0.0
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    current_test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += labels.size(0)
                    correct_test += predicted.eq(labels).sum().item()
            
            # 테스트 결과 저장
            test_losses.append(current_test_loss / len(test_loader))
            test_accuracies.append(100. * correct_test / total_test)
            
            print(f'Epoch [{epoch+1}] Test Loss: {current_test_loss/len(test_loader):.4f} '
                  f'Test Acc: {100.*correct_test/total_test:.2f}%')
        else: # eval_freq가 1이 아닐 경우, 빈 값 추가하여 리스트 길이 맞추기
            test_losses.append(test_losses[-1] if len(test_losses) > 0 else 0.0) # 이전 값으로 채우거나 0.0
            test_accuracies.append(test_accuracies[-1] if len(test_accuracies) > 0 else 0.0) # 이전 값으로 채우거나 0.0
                
        scheduler.step()
    
    print("Training finished.")

    # 학습 결과 시각화
    plot_results(args.num_epochs, train_losses, train_accuracies, test_losses, test_accuracies, args.model, args.dataset)


def plot_results(num_epochs, train_losses, train_accuracies, test_losses, test_accuracies, model_name, dataset_name):
    epochs_range = range(1, num_epochs + 1)
    
    # Loss 그래프
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    # eval_freq에 따라 test_losses의 길이를 맞추기 위해 조정
    test_epochs_range = [i * args.eval_freq for i in range(len(test_losses))]
    plt.plot(test_epochs_range, test_losses, label='Validation Loss')
    plt.title(f'{model_name} on {dataset_name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(test_epochs_range, test_accuracies, label='Validation Accuracy')
    plt.title(f'{model_name} on {dataset_name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 그래프 저장
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_{dataset_name}_results.png'))
    plt.show() # 그래프를 화면에 표시

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model: resnet34, densenet, fractalnet, preactresnet")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30, help="LR scheduler step")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--eval_freq", type=int, default=1, help="evaluation frequency")
    
    args = parser.parse_args()
    main(args)
