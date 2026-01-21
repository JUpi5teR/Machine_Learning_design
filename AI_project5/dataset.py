import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import *

# ===================== 数据增强（小样本专用） =====================
# 训练集增强：覆盖眼底图像拍摄差异
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # 模拟拍摄角度偏移
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.2), ratio=(0.9, 1.1)),  # 缩放裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])

# 验证/测试集：无增强，仅尺寸归一化
val_test_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# CLAHE增强（提升血管对比度，眼底图像专用）
def clahe_enhance(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return Image.fromarray(enhanced)


# ===================== 数据集类 =====================
class FundusVesselDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, is_train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train
        # 按数字排序（1.jpg,2.jpg...），确保标签匹配
        self.image_names = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 读取图像（转灰度+CLAHE增强）
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')  # 转灰度
        image = clahe_enhance(image)  # 增强血管对比度

        # 训练集：读取标签并处理
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, img_name)
            label = Image.open(label_path).convert('L')  # 标签为灰度图
            # 标签二值化：血管→1，背景→0（后续预测后反转回0/255）
            label = transforms.Resize(INPUT_SIZE)(label)
            label = np.array(label)
            label = (label == 0).astype(np.float32)  # 原始标签血管=0 → 模型训练用1
            label = torch.from_numpy(label).unsqueeze(0)  # [1, H, W]

        # 应用transform
        if self.transform:
            image = self.transform(image)

        # 返回：训练集（图像，标签，文件名）| 测试集（图像，文件名）
        if self.label_dir is not None:
            return image, label, img_name
        else:
            return image, img_name


# ===================== 数据加载器 =====================
def get_dataloaders():
    # 完整训练集
    full_dataset = FundusVesselDataset(
        TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, train_transform, is_train=True
    )
    # 拆分训练/验证集
    val_size = int(VALID_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 替换验证集transform（无增强）
    val_dataset.dataset.transform = val_test_transform

    # 测试集
    test_dataset = FundusVesselDataset(
        TEST_IMAGE_DIR, label_dir=None, transform=val_test_transform, is_train=False
    )

    # 加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader