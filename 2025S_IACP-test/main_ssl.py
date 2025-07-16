import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

# 사용자 정의 모듈
import models
import datasets
import frameworks

def plot_knn_accuracy(epochs, accuracies, eval_freq, args):
    #주기적으로 측정된 k-NN 정확도 그래프를 그리고 저장하는 함수
    plt.figure(figsize=(10, 6))
    
    # 평가가 수행된 에포크 지점을 x축으로 설정
    eval_epochs = range(eval_freq, epochs + 1, eval_freq)
    
    plt.plot(eval_epochs, accuracies, color='blue', marker='o', linestyle='-', label='k-NN Test Accuracy')
    
    plt.title(f'k-NN Accuracy over Epochs: {args.model} on {args.dataset}', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('k-NN Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 101)

    # 그래프 저장
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'ssl_{args.model}_{args.dataset}_knn_accuracy.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    
    print(f"\n k-NN accuracy plot saved to: {save_path}")
    plt.show()

def ssl_training(framework, train_loader, test_loader, train_labeled_loader, device, args):
    #SSL training loop with periodic k-NN evaluation
    optimizer = optim.SGD(framework.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    
    knn_accuracies = []
    
    for epoch in range(args.ssl_epochs):
        framework.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = framework.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = framework(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.ssl_epochs}] | Step [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{args.ssl_epochs}] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}')
        
        if (epoch + 1) % args.eval_freq == 0:
            print("\n" + "-"*25 + f" k-NN EVALUATION @ Epoch {epoch+1} " + "-"*25)
            train_features, train_labels = framework.collect_features(train_labeled_loader, device)
            test_features, test_labels = framework.collect_features(test_loader, device)
            
            if train_features is not None:
                accuracy = framework.knn_evaluation(train_features, train_labels, test_features, test_labels, k=5)
                knn_accuracies.append(accuracy * 100)
                print(f"Current Best k-NN Accuracy: {accuracy*100:.2f}%")
            print("-" * (62 + len(str(epoch+1))) + "\n")

        scheduler.step()
    
    return knn_accuracies

def main(args):
    num_classes = datasets.get_num_classes(args.dataset)
    encoder = models.load_model(args.model, num_classes=num_classes)
    framework = frameworks.RotNet(encoder)

    # 데이터셋 및 데이터로더 설정 (이전과 동일)
    train_dataset = datasets.load_dataset(args.dataset, train=True, ssl_mode=True)
    test_dataset = datasets.load_dataset(args.dataset, train=False, ssl_mode=False)
    train_labeled = datasets.load_dataset(args.dataset, train=True, ssl_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    framework.to(device)
    
    print("=" * 60)
    print("SELF-SUPERVISED LEARNING (RotNet)")
    print(f"Encoder: {args.model} | Dataset: {args.dataset} | Epochs: {args.ssl_epochs}")
    print("=" * 60)
    
    # SSL Training & Evaluation
    accuracies = ssl_training(framework, train_loader, test_loader, train_labeled_loader, device, args)
    
    print("SSL Training finished.")
    
    # 결과 그래프 생성
    if accuracies:
        plot_knn_accuracy(args.ssl_epochs, accuracies, args.eval_freq, args)
    else:
        print("No k-NN evaluation was performed, skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Supervised Learning with k-NN Accuracy Plotting')
    
    # 인자 설정 (이전과 동일)
    parser.add_argument("--model", type=str, default="resnet34", help="Encoder model")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Dataset")
    parser.add_argument("--ssl_epochs", type=int, default=100, help="SSL training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--print_freq", type=int, default=100, help="Print frequency")
    parser.add_argument("--eval_freq", type=int, default=10, help="k-NN evaluation frequency")
    parser.add_argument("--framework", type=str, default="rotnet", choices=["rotnet"], help="SSL framework")
    parser.add_argument("--num_blocks", type=int, default=3, choices=[3, 4, 5], help="NIN blocks for RotNet model")

    args = parser.parse_args()
    main(args)