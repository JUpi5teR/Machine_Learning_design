import os
import cv2
import torch
import subprocess
import shutil
import numpy as np
from tqdm import tqdm
from config import *
from model import init_model
from dataset import get_dataloaders
from utils import post_process

# 确保预测目录存在
os.makedirs(PRED_DIR_FOR_CSV, exist_ok=True)


def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        # 修正：DataLoader已添加批次维度，无需再unsqueeze
        pred = model(image_tensor)
        pred_mask = post_process(pred.squeeze(0))  # 后处理+转0/255
    return pred_mask


def batch_predict():
    model = init_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    _, _, test_loader = get_dataloaders()

    print(f"Predicting test images (save to {PRED_DIR_FOR_CSV} )...")
    for image_tensor, img_name in tqdm(test_loader):
        pred_mask = predict_image(model, image_tensor)
        save_path = os.path.join(PRED_DIR_FOR_CSV, img_name[0])
        # 确保保存成功（添加校验）
        if not cv2.imwrite(save_path, pred_mask):
            raise IOError(f"无法保存预测结果到 {save_path}")
    # 验证预测结果是否生成
    if not os.listdir(PRED_DIR_FOR_CSV):
        raise RuntimeError(f"{PRED_DIR_FOR_CSV} 目录下未生成预测图像，请检查预测逻辑")


def generate_submission_csv():
    original_cwd = os.getcwd()
    # 构造绝对路径确保脚本能找到预测目录
    pred_abs_dir = os.path.abspath(PRED_DIR_FOR_CSV)

    try:
        # 直接在当前目录调用脚本，并传递绝对路径参数
        subprocess.run(
            ["python", CSV_SCRIPT_PATH, pred_abs_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"成功调用 {CSV_SCRIPT_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"执行脚本出错: {e.stderr}")
        return
    except FileNotFoundError:
        print(f"未找到脚本 {CSV_SCRIPT_PATH}")
        return

    # 移动CSV到根目录
    csv_src = "./submission.csv"  # 假设脚本在当前目录生成CSV
    csv_dst = os.path.join(original_cwd, "submission.csv")
    if os.path.exists(csv_src):
        shutil.move(csv_src, csv_dst)
        print(f"CSV文件已保存到 {csv_dst}")
    else:
        print("未生成submission.csv，请检查segmentation_to_csv.py的输出路径")


if __name__ == "__main__":
    batch_predict()
    generate_submission_csv()