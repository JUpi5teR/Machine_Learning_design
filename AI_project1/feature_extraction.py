import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from scipy.stats import skew, kurtosis
from skimage.measure import regionprops


def extract_color_features(image):
    """提取颜色特征（固定长度）"""
    # 转换色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 固定分箱数的颜色直方图
    def get_hist_channel(channel, bins=32):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256 if channel.dtype == np.uint8 else 180])
        return cv2.normalize(hist, hist).flatten()

    # BGR通道直方图
    b, g, r = cv2.split(image)
    hist_b = get_hist_channel(b)
    hist_g = get_hist_channel(g)
    hist_r = get_hist_channel(r)

    # HSV通道直方图
    h, s, v = cv2.split(hsv)
    hist_h = get_hist_channel(h, bins=18)  # H通道范围0-179
    hist_s = get_hist_channel(s)
    hist_v = get_hist_channel(v)

    # LAB通道直方图
    l, a, b_channel = cv2.split(lab)
    hist_l = get_hist_channel(l)
    hist_a = get_hist_channel(a)
    hist_b_lab = get_hist_channel(b_channel)

    # 颜色矩特征（一阶到三阶矩）
    def color_moments(channel):
        mean = np.mean(channel)
        std = np.std(channel)
        sk = skew(channel.flatten())
        kurt = kurtosis(channel.flatten())
        return [mean, std, sk, kurt]

    # 计算所有通道的颜色矩
    moments = []
    for channel in [b, g, r, h, s, v, l, a, b_channel]:
        moments.extend(color_moments(channel))

    # 拼接所有颜色特征（固定长度：32*9 + 9*4 = 288 + 36 = 324）
    return np.concatenate([
        hist_b, hist_g, hist_r,
        hist_h, hist_s, hist_v,
        hist_l, hist_a, hist_b_lab,
        moments
    ])


def extract_texture_features(image):
    """提取纹理特征（固定长度）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 多尺度LBP特征
    lbp_features = []
    radii = [1, 2, 3]
    for r in radii:
        n_points = 8 * r
        lbp = local_binary_pattern(gray, n_points, r, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
        lbp_features.extend(hist / np.sum(hist))  # 归一化

    # 灰度共生矩阵特征（固定4个方向）
    def glcm_features(gray_img):
        from skimage.feature import graycomatrix, graycoprops
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray_img, distances, angles, 256, symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        features = []
        for prop in props:
            features.extend(graycoprops(glcm, prop).ravel())
        return features

    # 确保灰度图是uint8类型
    gray_uint8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm_feats = glcm_features(gray_uint8)

    # 图像熵和边缘特征
    hist_gray = np.histogram(gray, bins=256, range=(0, 256))[0]
    hist_gray = hist_gray / np.sum(hist_gray)
    entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-10))

    # Sobel边缘统计
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_stats = [np.mean(edge_mag), np.std(edge_mag), np.max(edge_mag)]

    # 拼接纹理特征（固定长度：(10+18+26) + 20 + 1 + 3 = 54 + 20 + 4 = 78）
    return np.concatenate([
        lbp_features,  # 多尺度LBP (10+18+26)
        glcm_feats,  # GLCM特征 (20)
        [entropy],  # 熵 (1)
        edge_stats  # 边缘统计 (3)
    ])


def extract_shape_features(image):
    """提取形状特征（固定长度，不依赖图像尺寸）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应阈值分割
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 形状特征（固定长度）
    shape_feats = []
    if contours:
        # 选择面积最大的轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # 形状不变量（与尺寸无关）
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            compactness = np.sqrt(4 * area / np.pi) / perimeter if perimeter > 0 else 0
        else:
            circularity = 0
            compactness = 0

        # 边界框特征（比例特征与尺寸无关）
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0

        # 椭圆拟合特征
        if len(largest) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(largest)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            eccentricity = 0

        # 基于区域的特征
        props = regionprops(binary.astype(np.int32))[0] if regionprops(binary.astype(np.int32)) else None
        solidity = props.solidity if props else 0
        extent = props.extent if props else 0

        shape_feats = [
            circularity, compactness, aspect_ratio,
            extent, solidity, eccentricity,
            area / (gray.shape[0] * gray.shape[1])  # 归一化面积（与图像尺寸无关）
        ]
    else:
        # 如果没有检测到轮廓，用0填充
        shape_feats = [0.0] * 7

    # HOG特征（固定长度处理）
    def get_fixed_hog(gray_img, target_length=128):
        # 多尺度提取后池化到固定长度
        scales = [0.5, 1.0, 1.5]
        all_hog = []
        for s in scales:
            h = max(32, int(gray_img.shape[0] * s))  # 最小32像素
            w = max(32, int(gray_img.shape[1] * s))
            resized = cv2.resize(gray_img, (w, h))
            hog_feat = hog(
                resized, orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False
            )
            all_hog.append(hog_feat)

        # 合并并池化到固定长度
        combined = np.concatenate(all_hog)
        if len(combined) < target_length:
            return np.pad(combined, (0, target_length - len(combined)), mode='constant')
        else:
            indices = np.linspace(0, len(combined) - 1, target_length, dtype=int)
            return combined[indices]

    hog_feats = get_fixed_hog(gray)

    # 拼接形状特征（固定长度：7 + 128 = 135）
    return np.concatenate([shape_feats, hog_feats])


def extract_all_features(images):
    """提取所有特征并组合（确保固定长度）"""
    all_features = []
    expected_length = None  # 验证特征长度一致性

    for img in images:
        try:
            color_feat = extract_color_features(img)
            texture_feat = extract_texture_features(img)
            shape_feat = extract_shape_features(img)

            # 组合所有特征
            combined_feat = np.concatenate([color_feat, texture_feat, shape_feat])

            # 验证特征长度
            if expected_length is None:
                expected_length = len(combined_feat)
            else:
                assert len(combined_feat) == expected_length, \
                    f"特征长度不一致: {len(combined_feat)} != {expected_length}"

            all_features.append(combined_feat)
        except Exception as e:
            print(f"特征提取错误: {e}")
            # 使用零向量替换错误特征（保持长度一致）
            all_features.append(np.zeros(expected_length) if expected_length else None)

    # 过滤可能的None值
    all_features = [f for f in all_features if f is not None]
    return np.array(all_features)