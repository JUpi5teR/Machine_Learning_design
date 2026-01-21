import torch
import torch.optim as optim
from tqdm import tqdm
from config import *
from model import init_model
from dataset import get_dataloaders
from utils import CombinedLoss, EarlyStopping, dice_coeff  # 现在不会报错了


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coeff(outputs, labels).item()
        pbar.set_postfix({'L': f'{loss.item():.3f}', 'D': f'{total_dice / len(loader):.3f}'})
    return total_loss / len(loader), total_dice / len(loader)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_dice += dice_coeff(outputs, labels).item()
    return total_loss / len(loader), total_dice / len(loader)


if __name__ == "__main__":
    # 使用修改后的 AttentionResUNet 模型
    model = init_model(pretrained=False)
    criterion = CombinedLoss()
    # 增加 weight_decay 提高模型学习的泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=PATIENCE)
    train_loader, val_loader, _ = get_dataloaders()

    best_dice = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_l, train_d = train_epoch(model, train_loader, criterion, optimizer)
        val_l, val_d = val_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"Val Loss: {val_l:.4f} | Val Dice: {val_d:.4f}")

        if val_d > best_dice:
            best_dice = val_d
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved Best Model: {best_dice:.4f}")

        if early_stopping(val_d):
            print("Early stopping!")
            break