import os
import cv2
import numpy as np
from PIL import Image


def load_images_from_directory(data_dir, is_train=True):
    """加载图像数据（保持原始尺寸）"""
    # 类别名称定义
    CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]

    images = []
    labels = []
    filenames = []
    # 移除强制resize，保持原始尺寸
    # target_size = (128, 128)

    if is_train:
        # 处理训练集
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # 加载图像（不改变尺寸）
                        img = cv2.imread(img_path)
                        if img is None:
                            img = np.array(Image.open(img_path).convert('RGB'))
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # 不进行resize，保持原始尺寸
                        # img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(class_idx)
                        filenames.append(img_name)
                    except Exception as e:
                        print(f"加载图像失败 {img_path}: {e}")
    else:
        # 处理测试集
        for img_name in os.listdir(data_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        img = np.array(Image.open(img_path).convert('RGB'))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # 不进行resize，保持原始尺寸
                    # img = cv2.resize(img, target_size)
                    images.append(img)
                    filenames.append(img_name)
                except Exception as e:
                    print(f"加载图像失败 {img_path}: {e}")

    return images, labels, filenames