import os
import torch

# 路径配置
DATA_ROOT = "dataset-for-task2"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
MODEL_SAVE_PATH = "best_plant_model.pth"
OUTPUT_CSV_PATH = "submission.csv"

# 超参数配置（针对过拟合优化）
BATCH_SIZE = 16  # 减小批次大小增加随机性
LEARNING_RATE = 0.0001  # 降低学习率减缓过拟合
EPOCHS = 30  # 增加训练轮数配合早停
NUM_CLASSES = 5
VAL_SPLIT = 0.3  # 增加验证集比例
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 1e-4  # 新增权重衰减参数

# 图片预处理参数
IMG_SIZE = 224  # 保持ResNet标准输入尺寸