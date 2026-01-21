import torch
import pandas as pd
import config
from model import get_model
from dataset import get_test_loader
import os

def predict():
    # 1. 准备环境
    device = torch.device(config.DEVICE)
    model = get_model()
    # 加载训练好的权重
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 2. 类别映射（严格与训练时ImageFolder生成的顺序一致）
    class_names = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
    print(f"Class mapping for prediction: {class_names}")

    # 3. 加载测试数据并推理
    test_loader = get_test_loader()
    ids = []
    categories = []
    print("Starting prediction...")

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)
            # 转换为类别名称
            predicted_indices = predicted_indices.cpu().numpy()
            for i, idx in enumerate(predicted_indices):
                ids.append(filenames[i])
                categories.append(class_names[idx])

    # 4. 生成并保存CSV
    df = pd.DataFrame({
        'ID': ids,
        'Category': categories
    })
    df = df.sort_values(by='ID', ascending=True)  # 按ID排序
    df = df.reset_index(drop=True)
    df.to_csv(config.OUTPUT_CSV_PATH, index=False)
    print(f"Prediction complete. File saved to {config.OUTPUT_CSV_PATH}")
    print("\nPreview of generated CSV:")
    print(df.head())


if __name__ == "__main__":
    predict()