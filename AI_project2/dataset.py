import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import config

# 定义数据预处理（增强与标准化）
# 训练集：增强策略强化
train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),  # 扩大旋转角度
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 新增仿射变换
    transforms.RandomCrop(config.IMG_SIZE, padding=16),  # 新增随机裁剪
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 增强颜色抖动
    transforms.RandomGrayscale(p=0.1),  # 新增随机灰度
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证集和测试集：无增强，仅基础预处理
val_test_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_train_val_loader():
    """从训练集拆分验证集，返回训练/验证加载器和类别映射"""
    # 加载完整训练集
    full_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transforms)
    # 计算拆分比例
    val_size = int(config.VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    # 随机拆分
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 验证集使用无增强的预处理
    val_dataset.dataset.transform = val_test_transforms
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader, full_dataset.class_to_idx


def get_test_loader():
    test_dataset = TestDataset(root_dir=config.TEST_DIR, transform=val_test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    return test_loader