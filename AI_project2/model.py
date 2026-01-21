import torch.nn as nn
from torchvision import models
import config


def get_model():
    # 使用预训练的ResNet18作为基础模型
    model = models.resnet18(pretrained=True)

    # 冻结大部分预训练层（只微调最后几层）
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层为带批归一化和dropout的结构
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),  # 批归一化稳定训练
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout抑制过拟合
        nn.Linear(512, config.NUM_CLASSES)
    )

    # 解冻最后一个卷积块和分类头
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(config.DEVICE)