import os
import shutil
import numpy as np
import yaml
from lxml import etree
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import config


def parse_xml_to_yolo(xml_path, img_w, img_h):
    """将XML bbox转换为YOLO归一化格式 [x_center, y_center, w, h]"""
    if not os.path.exists(xml_path):
        return None

    tree = etree.parse(xml_path)
    root = tree.getroot()
    bbox = []

    for obj in root.findall("object"):
        if obj.find("name").text == "Fovea":
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 转换为 xywh (归一化)
            dw = 1. / img_w
            dh = 1. / img_h
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2.0
            y_center = ymin + h / 2.0

            w = w * dw
            h = h * dh
            x_center = x_center * dw
            y_center = y_center * dh

            # 类别 0 为 Fovea
            return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

    return None


def prepare_yolo_dataset():
    """
    准备YOLO格式的数据集结构：
    yolo_dataset_cache/
        images/train, images/val
        labels/train, labels/val
        data.yaml
    注意：为了节省磁盘空间，images文件夹内使用硬链接或复制
    """
    print("正在构建 YOLO 数据集格式...")

    # 定义目录
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(config.YOLO_DATASET_DIR, d), exist_ok=True)

    # 获取所有图片
    img_files = [f for f in os.listdir(config.TRAIN_IMG_DIR) if f.endswith(('.jpg', '.png'))]
    train_files, val_files = train_test_split(img_files, test_size=0.15, random_state=42)

    def process_files(files, split_name):
        for img_name in tqdm(files, desc=f"Processing {split_name}"):
            src_img_path = os.path.join(config.TRAIN_IMG_DIR, img_name)
            xml_name = img_name.replace(".jpg", ".xml").replace(".png", ".xml")
            src_xml_path = os.path.join(config.TRAIN_XML_DIR, xml_name)

            # 1. 复制/链接图片
            dst_img_path = os.path.join(config.YOLO_DATASET_DIR, 'images', split_name, img_name)
            if not os.path.exists(dst_img_path):
                shutil.copy(src_img_path, dst_img_path)  # 使用copy保证兼容性

            # 2. 生成 Label
            if os.path.exists(src_img_path):  # 读取图片尺寸
                import cv2
                img = cv2.imread(src_img_path)
                h, w = img.shape[:2]

                yolo_line = parse_xml_to_yolo(src_xml_path, w, h)

                label_name = img_name.rsplit('.', 1)[0] + ".txt"
                dst_label_path = os.path.join(config.YOLO_DATASET_DIR, 'labels', split_name, label_name)

                with open(dst_label_path, 'w') as f:
                    if yolo_line:
                        f.write(yolo_line)
                    # 如果没有bbox，生成空文件表示背景（YOLO支持）

    process_files(train_files, 'train')
    process_files(val_files, 'val')

    # 生成 data.yaml
    yaml_content = {
        'path': os.path.abspath(config.YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'Fovea'}
    }

    yaml_path = os.path.join(config.YOLO_DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"YOLO 数据集准备完毕: {yaml_path}")
    return yaml_path