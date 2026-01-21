import torch
import os

# 路径配置保持不变...
TRAIN_IMAGE_DIR = "./segmentation/train/image"
TRAIN_LABEL_DIR = "./segmentation/train/label"
TEST_IMAGE_DIR = "./segmentation/test/image"
MODEL_SAVE_PATH = "./best_attention_resunet.pth"
PRED_DIR_FOR_CSV = "./segmentation/test/pred"
os.makedirs(PRED_DIR_FOR_CSV, exist_ok=True)

# 训练配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (512, 512)
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
HIDDEN_CHANNELS = 32
BATCH_SIZE = 4         # 增加 Batch Size 提高稳定性
LEARNING_RATE = 2e-4   # 配合 AdamW
EPOCHS = 60
PATIENCE = 12
VALID_SPLIT = 0.2

# 关键：下调 POS_WEIGHT。
# 既然 Dice 已经 0.96，不需要 10 倍惩罚，降到 3.0-5.0 会让 Loss 数值立刻变小
POS_WEIGHT = 4.0