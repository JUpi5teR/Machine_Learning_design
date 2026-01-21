import os
import numpy as np
import joblib

def predict_with_model(features, model_path="models/best_model.pkl",
                      scaler_path="models/scaler.pkl",
                      selector_path="models/feature_selector.pkl"):
    """使用训练好的模型进行预测"""
    # 加载模型和预处理工具
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    if not os.path.exists(selector_path):
        raise FileNotFoundError(f"特征选择器文件不存在: {selector_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    # 预处理特征
    features_scaled = scaler.transform(features)
    features_selected = selector.transform(features_scaled)

    # 预测
    predictions = model.predict(features_selected)
    probabilities = model.predict_proba(features_selected)

    # 类别名称映射
    class_names = ["Black-grass", "Common wheat", "Loose Silky-bent",
                   "Scentless Mayweed", "Sugar beet"]

    # 转换预测结果
    pred_labels = [class_names[int(pred)] for pred in predictions]
    return pred_labels, probabilities

def save_predictions(filenames, predictions, probabilities, output_path="submission-for-task1.csv"):
    """保存预测结果到CSV文件（适配目标格式：仅ID和Category）"""
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头（与参考文件一致：ID和Category）
        writer.writerow(["ID", "Category"])
        # 仅写入文件名（ID）和预测类别（Category）
        for filename, pred in zip(filenames, predictions):
            writer.writerow([filename, pred])
    print(f"预测结果已保存到: {output_path}")