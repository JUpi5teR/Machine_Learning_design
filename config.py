import os
import torch


class Config:
    # 项目根目录（当前文件所在目录）
    _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = _PROJECT_DIR  # 使用当前项目目录作为根目录
    
    # 数据路径（相对于项目根目录）
    TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train")
    TRAIN_XML_DIR = os.path.join(ROOT_DIR, "train_location")
    TEST_IMG_DIR = os.path.join(ROOT_DIR, "test")
    
    # CSV文件路径（如果文件不存在，会在需要时创建）
    GT_CSV_PATH = os.path.join(ROOT_DIR, "fovea_localization_train_GT.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(ROOT_DIR, "sample_submission.csv")

    # 输出路径
    SAVE_MODEL_DIR = os.path.join(_PROJECT_DIR, "saved_models")
    # YOLO 训练数据缓存路径 (自动生成，保存在项目目录下)
    YOLO_DATASET_DIR = os.path.join(_PROJECT_DIR, "yolo_dataset_cache")

    # 训练超参数 (针对 YOLO 调整，优化MSE，适配8GB显存)
    BATCH_SIZE = 8  # 降低batch size以适应显存限制
    EPOCHS = 250  # 进一步增加训练轮数以获得更好的收敛
    LR0 = 0.0008  # 更精细的初始学习率
    IMG_SIZE = 640  # 使用640尺寸以适应8GB显存（如果显存充足可以增加到832）
    MULTI_SCALE_TRAINING = True  # 启用多尺度训练（提高泛化能力）
    
    # 推理优化参数（高级优化）
    INFERENCE_IMG_SIZE = 640  # 推理时使用640尺寸（如果显存充足可以增加到832）
    USE_TTA = True  # 使用测试时增强（Test Time Augmentation）
    TTA_SCALES = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]  # 更多尺度以提高精度
    TTA_FLIPS = True  # 启用翻转增强
    USE_CONFIDENCE_WEIGHTED = True  # 使用置信度加权平均
    CONF_THRESHOLD = 0.001  # 降低置信度阈值，确保能检测到目标
    POST_PROCESS_SMOOTH = True  # 启用后处理平滑（对连续图像的结果进行平滑）

    # 模型配置 - 使用中等模型以平衡精度和显存
    # yolo11n < yolo11s < yolo11m < yolo11l < yolo11x
    MODEL_NAME = "yolo11m.pt"  # 使用中等模型以适应显存限制（如果显存充足可以改为yolo11l.pt）

    @property
    def device_str(self):
        """返回YOLO训练使用的设备字符串"""
        if self.DEVICE == "cuda":
            return str(self.GPU_ID)  # YOLO接受数字字符串，如 "0" 或 "0,1" 用于多GPU
        return "cpu"
    
    @property
    def device_for_inference(self):
        """返回推理时使用的设备"""
        if self.DEVICE == "cuda":
            return f"cuda:{self.GPU_ID}"
        return "cpu"


def _get_device_config():
    """获取GPU设备配置，支持多GPU"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_id = 0  # 默认使用第一块GPU，可以通过环境变量或参数修改
        
        # 显示GPU信息
        print(f"✅ 检测到 {gpu_count} 块GPU:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            current_device = " (当前使用)" if i == gpu_id else ""
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB){current_device}")
        
        return "cuda", gpu_id
    else:
        print("⚠️  未检测到CUDA，将使用CPU（速度较慢）")
        return "cpu", None


# 初始化配置实例
config = Config()
# 初始化设备配置
config.DEVICE, config.GPU_ID = _get_device_config()
os.makedirs(config.SAVE_MODEL_DIR, exist_ok=True)