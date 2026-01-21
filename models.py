from ultralytics import YOLO
from config import config
import os
import torch


def get_yolo_model(model_path=None):
    """
    加载YOLO模型并移动到指定设备。
    如果是第一次运行，会下载预训练权重。
    """
    if model_path:
        model = YOLO(model_path)
    else:
        # 默认加载 config 中定义的模型 (例如 yolo11s.pt)
        model = YOLO(config.MODEL_NAME)
    
    # 确保模型在正确的设备上
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        # YOLO会自动处理设备，但我们可以显式设置
        print(f"模型将使用GPU: {torch.cuda.get_device_name(config.GPU_ID)}")
    else:
        print("模型将使用CPU")
    
    return model