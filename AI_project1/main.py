import os
import sys

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)


def main():
    # 1. 定义路径
    train_dir = os.path.join(PROJECT_ROOT, "dataset-for-task1", "train")
    test_dir = os.path.join(PROJECT_ROOT, "dataset-for-task1", "test")
    model_dir = os.path.join(PROJECT_ROOT, "models")

    # 2. 加载数据（不改变原始尺寸）
    print("=" * 50)
    print("加载训练数据...")
    from data_loading import load_images_from_directory
    train_images, train_labels, train_filenames = load_images_from_directory(train_dir, is_train=True)
    print(f"训练集加载完成: {len(train_images)} 张图片")

    print("\n加载测试数据...")
    test_images, _, test_filenames = load_images_from_directory(test_dir, is_train=False)
    print(f"测试集加载完成: {len(test_images)} 张图片")

    # 3. 特征提取
    print("\n" + "=" * 50)
    print("提取训练集特征...")
    from feature_extraction import extract_all_features
    train_features = extract_all_features(train_images)
    print(f"训练集特征提取完成: {train_features.shape}")

    print("\n提取测试集特征...")
    test_features = extract_all_features(test_images)
    print(f"测试集特征提取完成: {test_features.shape}")

    # 4. 模型训练
    print("\n" + "=" * 50)
    print("开始模型训练...")
    from model_training import train_models
    best_model, scaler, selector = train_models(train_features, train_labels, output_dir=model_dir)
    print("模型训练完成!")

    # 5. 测试集预测
    print("\n" + "=" * 50)
    print("开始测试集预测...")
    from prediction import predict_with_model, save_predictions
    predictions, probabilities = predict_with_model(
        test_features,
        model_path=os.path.join(model_dir, "best_model.pkl"),
        scaler_path=os.path.join(model_dir, "scaler.pkl"),
        selector_path=os.path.join(model_dir, "feature_selector.pkl")
    )

    # 6. 保存预测结果
    output_path = os.path.join(PROJECT_ROOT, "test_predictions.csv")
    save_predictions(test_filenames, predictions, probabilities, output_path)

    # 7. 显示预测统计
    print("\n" + "=" * 50)
    print("预测结果统计:")
    from collections import Counter
    pred_counts = Counter(predictions)
    for plant, count in pred_counts.items():
        print(f"{plant}: {count} 张图片")

    print("\n所有任务完成!")


if __name__ == "__main__":
    main()