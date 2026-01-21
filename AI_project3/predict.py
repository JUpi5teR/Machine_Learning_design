import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from config import *
from data_loader import load_test_data


def generate_submission():
    """
    加载测试集，进行预测，生成提交文件
    """
    # 1. 检查模型文件是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_SAVE_PATH} 不存在，请先运行train.py训练模型")

    # 2. 加载测试数据
    x_test, test_filenames = load_test_data()

    # 3. 加载训练好的模型
    print(f"加载模型：{MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)

    # 4. 进行预测
    print("开始预测测试集...")
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)

    # 5. 转换预测结果（从概率分布到类别标签）
    # 取概率最大的类别作为最终预测结果
    predicted_labels = np.argmax(predictions, axis=1)

    # 6. 生成提交DataFrame
    # 确保提交文件的格式与示例完全一致：两列（ID, Emotion）
    submission_df = pd.DataFrame({
        'ID': test_filenames,
        'Emotion': predicted_labels
    })

    # 7. 保存提交文件
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提交文件已生成：{SUBMISSION_PATH}")

    # 8. 显示提交文件的基本信息
    print(f"\n提交文件信息：")
    print(f"  - 总样本数：{len(submission_df)}")
    print(f"  - 情感标签分布：")
    label_distribution = submission_df['Emotion'].value_counts().sort_index()
    for label, count in label_distribution.items():
        emotion_name = EMOTION_MAPPING.get(label, "未知")
        print(f"    {label} ({emotion_name}): {count} 个样本")

    # 显示前10行结果
    print(f"\n提交文件前10行预览：")
    print(submission_df.head(10).to_string(index=False))

    return submission_df


if __name__ == "__main__":
    # 生成提交文件
    generate_submission()