import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *


def load_train_data():
    """
    加载并调整训练集尺寸
    """
    images = []
    labels = []

    print("正在加载训练数据...")
    for emotion_name, label in LABEL_MAPPING.items():
        emotion_dir = os.path.join(TRAIN_DIR, emotion_name)
        if not os.path.exists(emotion_dir):
            continue

        for img_filename in os.listdir(emotion_dir):
            if img_filename.endswith(".jpg"):
                img_path = os.path.join(emotion_dir, img_filename)

                # 读取灰度
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # resize 会自动读取 config.py 里的 (64, 64)
                img = cv2.resize(img, IMAGE_SIZE)

                images.append(img / 255.0)  # 归一化
                labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    images = np.expand_dims(images, axis=-1)  # (N, 64, 64, 1)

    # 划分验证集
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}")
    return x_train, x_val, y_train, y_val


def create_data_generators():
    """
    增强版数据生成器
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,  # 增加旋转角度
        width_shift_range=0.2,  # 增加偏移容忍度
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()  # 验证集不做增强
    return train_datagen, val_datagen


def load_test_data():
    """
    加载测试集
    """
    test_images = []
    test_filenames = []

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError("Test dir not found")

    print("正在加载测试数据...")
    for img_filename in os.listdir(TEST_DIR):
        if img_filename.endswith(".jpg"):
            img_path = os.path.join(TEST_DIR, img_filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMAGE_SIZE)  # 确保测试集也是 64x64

            test_images.append(img / 255.0)
            test_filenames.append(img_filename)

    test_images = np.array(test_images, dtype=np.float32)
    test_images = np.expand_dims(test_images, axis=-1)

    return test_images, test_filenames