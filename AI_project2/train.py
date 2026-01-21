import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_train_val_loader
from model import get_model
import os


def train():
    print(f"Using device: {config.DEVICE}")

    # 1. 加载数据（包含训练集和验证集）
    train_loader, val_loader, class_mapping = get_train_val_loader()
    print(f"Classes found: {class_mapping}")

    # 2. 初始化模型、损失函数、优化器
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    # 优化器添加权重衰减
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # 学习率调度器（验证损失不下降时降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 3. 训练循环（带验证集评估）
    best_val_loss = float('inf')
    patience = 5  # 增加早停耐心至5轮
    no_improve_epochs = 0

    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 20)

        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算训练集 epoch 指标
        train_epoch_loss = train_running_loss / train_total
        train_epoch_acc = train_correct / train_total
        print(f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}")

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证集 epoch 指标
        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

        # 更新学习率调度器
        scheduler.step(val_epoch_loss)

        # 早停机制与模型保存
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Model saved (best val loss updated)!")
            no_improve_epochs = 0  # 重置计数器
        else:
            no_improve_epochs += 1
            print(f"No improvement in val loss for {no_improve_epochs} epoch(s)")
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Class mapping (for prediction): {class_mapping}")


if __name__ == "__main__":
    train()