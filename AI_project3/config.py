import os

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "fer_data")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "emotion_model.h5")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ================= 标签映射 =================
EMOTION_MAPPING = {
    0: "Angry", 1: "Fear", 2: "Happy", 3: "Sad", 4: "Surprise", 5: "Neutral"
}
LABEL_MAPPING = {v: k for k, v in EMOTION_MAPPING.items()}

# ================= 关键参数调整 =================
# 1. 尺寸改为 64x64，这对于预训练模型提取特征至关重要（48x48太小会导致特征在深层消失）
IMAGE_SIZE = (64, 64)
# 2. 批量大小：32或64。如果显存报错，改回16
BATCH_SIZE = 32
# 3. 迁移学习可以在较少轮次内收敛
EPOCHS = 30
# 4. 初始学习率：预训练模型可以使用稍大的学习率
LEARNING_RATE = 1e-3
NUM_CLASSES = 6
# 5. 输入形状调整为 64x64
INPUT_SHAPE = (64, 64, 1)