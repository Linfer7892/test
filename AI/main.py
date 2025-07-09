import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 사용자 정의 모듈 (실행 환경에 맞게 준비되어 있어야 함)
import models
import datasets

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

    print("-----------------------------Hyper Parameter-----------------------------")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Num of Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print("-------------------------------------------------------------------------")
    print("Training start.")
    
    # 결과를 저장할 리스트 초기화
    train_accuracies = []
    test_accuracies = []

    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # print_freq마다 출력
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{args.num_epochs}] '
                      f'Step: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*correct_train/total_train:.2f}%')
        
        # 에포크 종료 후 학습 정확도 저장
        train_accuracies.append(100. * correct_train / total_train)

        # eval_freq마다 평가
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += labels.size(0)
                    correct_test += predicted.eq(labels).sum().item()
            
            # 테스트 정확도 저장
            current_test_acc = 100. * correct_test / total_test
            test_accuracies.append(current_test_acc)
            
            print(f'Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f} '
                  f'Test Acc: {current_test_acc:.2f}%')
        
        # 평가하지 않는 에포크의 경우, 이전 정확도 값으로 채워넣어 리스트 길이를 맞춤
        elif len(test_accuracies) > 0:
            test_accuracies.append(test_accuracies[-1])
            
        scheduler.step()
    
    print("Training finished.")

    # 학습 결과 시각화
    plot_accuracy(args.num_epochs, train_accuracies, test_accuracies, args.model, args.dataset)


def plot_accuracy(num_epochs, train_accuracies, test_accuracies, model_name, dataset_name):
    """학습 및 테스트 정확도를 시각화하고 파일로 저장하는 함수"""
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(epochs_range, test_accuracies, 'o-', label='Validation Accuracy')
    plt.title(f'{model_name} on {dataset_name} - Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100) # Y축 범위를 0-100%로 고정
    
    # 그래프 저장
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f'{model_name}_{dataset_name}_accuracy.png')
    plt.savefig(save_path)
    print(f"Accuracy plot saved to {save_path}")
    
    plt.show() # 그래프를 화면에 표시

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10/100 Training")
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model: resnet18, densenet, fractalnet, preactresnet18")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        help="Dataset: cifar10 or cifar100")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=30, help="LR scheduler step")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--eval_freq", type=int, default=1, help="evaluation frequency")
    
    args = parser.parse_args()
    main(args)
