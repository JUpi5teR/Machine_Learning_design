import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D, BatchNormalization
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from config import *


def build_emotion_model():
    """
    构建模型：修复了权重加载的层数不匹配问题
    """
    # 1. 定义模型的整体输入 (64, 64, 1)
    inputs = Input(shape=INPUT_SHAPE)

    # 2. 手动将 1通道 转为 3通道 (RGB)
    # 这一步是为了适配 EfficientNet 的输入要求
    x = Conv2D(3, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    # -----------------------------------------------------------
    # 关键修改：将 Base Model 的构建与权重加载解耦
    # -----------------------------------------------------------

    # 3. 创建标准的 EfficientNetB0 骨架
    # 注意：这里先不加载权重 (weights=None)，也不连接到前面的 x (input_tensor=None)
    # 我们让它作为一个独立的模块存在，输入形状设为标准的 (64, 64, 3)
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(64, 64, 3)
    )

    # 4. 安全加载权重
    local_weights_path = os.path.join(CURRENT_DIR, "efficientnetb0_notop.h5")

    if os.path.exists(local_weights_path):
        print(f"正在加载本地权重：{local_weights_path}")
        try:
            # key point: by_name=True 允许忽略层数差异，只加载名字匹配的层
            # skip_mismatch=True 跳过形状不匹配的层（如果有的话）
            base_model.load_weights(local_weights_path, by_name=True, skip_mismatch=True)
            print(">> 权重加载成功！(忽略了层数不匹配)")
        except Exception as e:
            print(f"权重加载警告：{e}")
            print("尝试不使用 skip_mismatch 参数加载...")
            base_model.load_weights(local_weights_path, by_name=True)
    else:
        print("未找到本地权重，尝试在线下载 'imagenet' 权重...")
        # 如果本地没有文件，只能尝试在线加载标准权重
        # 为了避免重新构建模型，这里我们重新实例化一个带权重的 base_model
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(64, 64, 3)
        )

    # 解冻模型，允许微调
    base_model.trainable = True

    # 5. 将 base_model 当作一层，连接到前面的 x
    # 此时 x 是 (Batch, 64, 64, 3)，正好符合 base_model 的输入
    x = base_model(x)

    # 6. 添加分类头
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # 7. 构建最终模型
    model = Model(inputs, outputs, name="Fixed_EfficientNet_Model")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model