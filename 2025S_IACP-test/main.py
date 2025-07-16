import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 이전에 제공된 사용자 정의 모듈
import models
import datasets

def get_training_config(model_name):
    """논문 명세에 따른 학습 설정 가져오기"""
    configs = {
        "fractalnet": {
            "lr": 0.02,
            "batch_size": 100,
            "num_epochs": 400,
            "lr_step": 200,
            "lr_gamma": 0.1
        },
        "densenet": {
            "lr": 0.1,
            "batch_size": 64,
            "num_epochs": 300,
            "lr_step": 150,
            "lr_gamma": 0.1
        },
        "default": {
            "lr": 0.1,
            "batch_size": 64,
            "num_epochs": 200,
            "lr_step": 100,
            "lr_gamma": 0.1
        }
    }
    return configs.get(model_name, configs["default"])

def plot_accuracy_graph(epochs, train_accs, test_accs, args):
    #학습 및 테스트 정확도 그래프를 그리고 저장하는 함수
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, epochs + 1)
    
    plt.plot(epoch_range, train_accs, 'o-', label='Training Accuracy')
    plt.plot(epoch_range, test_accs, 'o-', label='Test Accuracy')
    
    plt.title(f'Model: {args.model} on {args.dataset} ({args.optimizer.upper()})', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 101) # Y축 범위를 0~101%로 고정
    
    # 그래프 저장
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'{args.model}_{args.dataset}_{args.optimizer}_accuracy.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    
    print(f"\n Accuracy plot saved to: {save_path}")
    plt.show()

def main(args):
    # 모델 및 데이터셋 로드
    num_classes = datasets.get_num_classes(args.dataset)
    model = models.load_model(args.model, num_classes=num_classes)
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)

    # 논문 기반 설정 + 사용자 입력 오버라이드
    config = get_training_config(args.model)
    final_lr = args.lr if args.lr != 0.1 else config["lr"]
    final_batch_size = args.batch_size if args.batch_size != 64 else config["batch_size"]
    final_epochs = args.num_epochs if args.num_epochs != 200 else config["num_epochs"]
    final_lr_step = args.lr_step if args.lr_step != 100 else config["lr_step"]
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer 선택
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=final_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=final_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=final_lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=final_lr_step, gamma=args.lr_gamma)
    
    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)
    print(f"Model: {args.model}, Dataset: {args.dataset}, Optimizer: {args.optimizer.upper()}")
    print(f"Epochs: {final_epochs}, Batch Size: {final_batch_size}, Learning Rate: {final_lr}")
    print("=" * 70)
    
    # 정확도 저장을 위한 리스트
    train_accuracies = []
    test_accuracies = []

    # 학습 루프
    for epoch in range(final_epochs):
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
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch: [{epoch+1}/{final_epochs}] | Step: [{batch_idx+1}/{len(train_loader)}] | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct_train/total_train:.2f}%')
        
        # 에포크 학습 정확도 기록
        train_accuracies.append(100. * correct_train / total_train)
        
        # 평가
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_loss, correct_test, total_test = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += labels.size(0)
                    correct_test += predicted.eq(labels).sum().item()
            
            current_test_acc = 100. * correct_test / total_test
            test_accuracies.append(current_test_acc)
            print(f'Epoch [{epoch+1}] Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {current_test_acc:.2f}%')
        else:
            # 평가를 건너뛴 에포크는 이전 정확도 값을 그대로 사용
            if test_accuracies:
                test_accuracies.append(test_accuracies[-1])
            else:
                test_accuracies.append(0) # 첫 에포크부터 평가 안 할 경우
        
        scheduler.step()
    
    print("Training finished.")

    # 결과 그래프 생성 및 저장
    plot_accuracy_graph(final_epochs, train_accuracies, test_accuracies, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised Learning with Plotting')
    
    # 기본 인자
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--dataset", type=str, default="cifar10")
    
    # Optimizer 선택 인자
    parser.add_argument("--optimizer", type=str, default="sgd")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr_step", type=int, default=100, help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    
    # 로그 및 평가 주기
    parser.add_argument("--print_freq", type=int, default=50, help="Print frequency")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency")
    
    args = parser.parse_args()
    main(args)