import numpy as np
import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from config import *
from data_loader import load_train_data, create_data_generators
from model_builder import build_emotion_model


def train_model():
    # 1. 加载数据
    x_train, x_val, y_train, y_val = load_train_data()
    train_datagen, val_datagen = create_data_generators()

    # 2. 构建迁移学习模型
    model = build_emotion_model()

    # 3. 定义激进的回调策略
    callbacks = [
        # 只保存验证集准确率最高的模型
        ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1
        ),

        # 快速调整学习率：如果验证集 Loss 3个 epoch 不降，学习率直接乘 0.2
        # 这能让模型在卡住时快速寻找新的最优解
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3,
            min_lr=1e-6, verbose=1
        ),

        # 早停：如果准确率 8 个 epoch 都没有提升，就停止训练
        EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1
        )
    ]

    # 4. 训练
    print("\n=== 开始迁移学习训练 ===")
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    np.save(os.path.join(OUTPUT_DIR, "training_history.npy"), history.history)
    print(f"\n训练完成！最佳模型已保存至：{MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train_model()