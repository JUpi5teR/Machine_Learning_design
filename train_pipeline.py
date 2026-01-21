import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_utils import prepare_yolo_dataset
from models import get_yolo_model


def train():
    # 1. 准备数据 (转换为YOLO格式)
    yaml_path = prepare_yolo_dataset()

    # 2. 加载模型
    print(f"Loading model: {config.MODEL_NAME} ...")
    model = get_yolo_model()

    # 3. 开始训练
    # YOLO 能够自动处理 amp, 优化器, 学习率调度等
    print(f"\n🚀 开始训练 (设备: {config.DEVICE.upper()})")
    if config.DEVICE == "cuda":
        print(f"   使用GPU: {config.device_str}")
    
    results = model.train(
        data=yaml_path,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.device_str,  # 使用配置的设备字符串
        project=config.SAVE_MODEL_DIR,
        name='yolo_fovea_run',
        exist_ok=True,  # 覆盖同名实验
        patience=40,  # 进一步增加早停耐心值
        lr0=config.LR0,
        lrf=0.005,  # 更小的最终学习率因子，更精细的收敛
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=5.0,  # 增加预热轮数，更稳定的训练开始
        cos_lr=True,  # 使用余弦学习率调度（更平滑的衰减）
        augment=True,  # 启用内置数据增强
        amp=True,  # 自动混合精度训练（GPU加速）
        workers=4 if config.DEVICE == "cuda" else 2,  # 降低workers数量以节省显存
        cache=False,  # 关闭缓存以节省显存（如果显存充足可以改为True）
        multi_scale=config.MULTI_SCALE_TRAINING,  # 多尺度训练
        # 优化数据增强策略
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=10.0,  # 旋转角度（增加旋转增强）
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=5.0,  # 剪切（增加剪切增强）
        perspective=0.0001,  # 透视变换
        flipud=0.0,  # 上下翻转（医学图像通常不需要）
        fliplr=0.5,  # 左右翻转
        mosaic=1.0,  # Mosaic增强
        mixup=0.1,  # Mixup增强（轻微）
        copy_paste=0.0,  # Copy-paste增强
        # 损失函数权重调整（针对小目标优化）
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # DFL损失权重
    )

    print(
        f"\n✅ 训练完成！最佳模型保存在: {os.path.join(config.SAVE_MODEL_DIR, 'yolo_fovea_run', 'weights', 'best.pt')}")


if __name__ == "__main__":
    train()