import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config


def batch_inference():
    # 1. 加载最佳模型
    best_model_path = os.path.join(config.SAVE_MODEL_DIR, 'yolo_fovea_run', 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到训练好的模型: {best_model_path}")

    print(f"加载模型进行推理: {best_model_path}")
    print(f"推理设备: {config.DEVICE.upper()}")
    if config.DEVICE == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(config.GPU_ID)}")
    
    model = YOLO(best_model_path)

    # 2. 准备图片列表
    test_img_paths = [
        os.path.join(config.TEST_IMG_DIR, f)
        for f in os.listdir(config.TEST_IMG_DIR)
        if f.endswith((".jpg", ".png"))
    ]
    # 按图片序号排序
    test_img_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

    if len(test_img_paths) != 20:
        raise ValueError(f"测试图像数量错误！需20张，实际{len(test_img_paths)}张")

    results = []
    predictions_list = []  # 用于后处理平滑

    # 3. 推理循环（使用TTA和多尺度推理优化）
    print("开始推理...")
    if config.USE_TTA:
        print(f"使用增强TTA：{len(config.TTA_SCALES)}个尺度" + 
              (f" + 翻转增强" if config.TTA_FLIPS else "") +
              (f" + 置信度加权" if config.USE_CONFIDENCE_WEIGHTED else ""))
    
    for img_path in tqdm(test_img_paths):
        img_basename = os.path.basename(img_path)
        img_num = int(img_basename.split(".")[0])

        if config.USE_TTA:
            # 使用增强TTA：多尺度推理、翻转增强、置信度加权
            # 读取原始图像尺寸用于坐标转换
            img = cv2.imread(img_path)
            orig_h, orig_w = img.shape[:2]
            
            all_predictions = []
            all_confidences = []
            
            # 多尺度推理
            for scale in config.TTA_SCALES:
                tta_size = int(config.INFERENCE_IMG_SIZE * scale)
                preds = model.predict(
                    img_path,
                    imgsz=tta_size,
                    conf=config.CONF_THRESHOLD,
                    verbose=False,
                    device=config.device_str,
                    augment=True  # 启用推理时增强
                )[0]
                boxes = preds.boxes.xywh.cpu().numpy()
                confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) > 0 else np.array([])
                
                if len(boxes) > 0:
                    box = boxes[0]  # 取置信度最高的
                    conf = float(confs[0]) if len(confs) > 0 else 0.5
                    all_predictions.append([box[0], box[1]])
                    all_confidences.append(conf)
            
            # 翻转增强（如果启用）
            if config.TTA_FLIPS:
                # 水平翻转
                img_flipped = cv2.flip(img, 1)
                flipped_path = img_path.replace('.jpg', '_flipped_temp.jpg').replace('.png', '_flipped_temp.png')
                cv2.imwrite(flipped_path, img_flipped)
                
                for scale in config.TTA_SCALES[:3]:  # 只对部分尺度进行翻转推理以节省时间
                    tta_size = int(config.INFERENCE_IMG_SIZE * scale)
                    preds = model.predict(
                        flipped_path,
                        imgsz=tta_size,
                        conf=config.CONF_THRESHOLD,
                        verbose=False,
                        device=config.device_str,
                        augment=False
                    )[0]
                    boxes = preds.boxes.xywh.cpu().numpy()
                    confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) > 0 else np.array([])
                    
                    if len(boxes) > 0:
                        box = boxes[0]
                        conf = float(confs[0]) if len(confs) > 0 else 0.5
                        # 翻转回原始坐标
                        x_flipped = orig_w - box[0]
                        all_predictions.append([x_flipped, box[1]])
                        all_confidences.append(conf)
                
                # 清理临时文件
                if os.path.exists(flipped_path):
                    os.remove(flipped_path)
            
            if len(all_predictions) > 0:
                all_preds = np.array(all_predictions)
                if config.USE_CONFIDENCE_WEIGHTED and len(all_confidences) > 0:
                    # 使用置信度加权平均
                    weights = np.array(all_confidences)
                    weights = weights / (weights.sum() + 1e-8)  # 归一化权重
                    x_pred = float(np.average(all_preds[:, 0], weights=weights))
                    y_pred = float(np.average(all_preds[:, 1], weights=weights))
                else:
                    # 简单平均
                    x_pred = float(np.mean(all_preds[:, 0]))
                    y_pred = float(np.mean(all_preds[:, 1]))
            else:
                # 如果所有尺度都没检测到，使用图像中心
                x_pred, y_pred = orig_w / 2.0, orig_h / 2.0
        else:
            # 标准推理（不使用TTA）
            preds = model.predict(
                img_path,
                imgsz=config.INFERENCE_IMG_SIZE,
                conf=config.CONF_THRESHOLD,
                verbose=False,
                device=config.device_str
            )[0]

            boxes = preds.boxes.xywh.cpu().numpy()

            if len(boxes) > 0:
                # 取置信度最高的预测
                x_pred, y_pred = boxes[0][0], boxes[0][1]
            else:
                # 如果没检测到，取图片中心 (兜底策略)
                h, w = preds.orig_shape
                x_pred, y_pred = w / 2.0, h / 2.0

        predictions_list.append([x_pred, y_pred])
        results.append([f"{img_num}_Fovea_X", x_pred])
        results.append([f"{img_num}_Fovea_Y", y_pred])
    
    # 4. 后处理平滑（如果启用）
    if config.POST_PROCESS_SMOOTH and len(predictions_list) > 1:
        print("应用后处理平滑...")
        predictions_array = np.array(predictions_list)
        smoothed_predictions = np.zeros_like(predictions_array)
        
        # 使用移动平均平滑（窗口大小为3）
        window_size = 3
        for i in range(len(predictions_array)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions_array), i + window_size // 2 + 1)
            smoothed_predictions[i] = np.mean(predictions_array[start_idx:end_idx], axis=0)
        
        # 更新结果
        results = []
        for i, (img_path, _) in enumerate(zip(test_img_paths, predictions_list)):
            img_num = int(os.path.basename(img_path).split(".")[0])
            x_smooth, y_smooth = smoothed_predictions[i]
            results.append([f"{img_num}_Fovea_X", x_smooth])
            results.append([f"{img_num}_Fovea_Y", y_smooth])

    # 5. 生成提交文件
    sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    submit_df = pd.DataFrame(results, columns=sample_df.columns)

    if len(submit_df) != 40:
        raise ValueError(f"结果行数错误！需40行，实际{len(submit_df)}行")

    submit_df.to_csv("submission.csv", index=False)
    print("✅ 提交文件生成成功！")
    print(submit_df.head(4))


if __name__ == "__main__":
    batch_inference()