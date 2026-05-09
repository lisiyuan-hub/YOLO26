import os
import random

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(0)
np.random.seed(0)


class BCCDDataAugmentation:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()

    # 1. 空间翻转 (PIL Image 类型处理)
    # 针对血细胞无方向性的特点，同时随机进行水平和垂直翻转
    def random_flip(self, img, boxes):
        if len(boxes) > 0:
            # 水平翻转
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 1] = 1.0 - boxes[:, 1]  # YOLO格式下 x_center = 1 - x_center

            # 垂直翻转
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                boxes[:, 2] = 1.0 - boxes[:, 2]  # YOLO格式下 y_center = 1 - y_center
        return img, boxes

    # 2. 颜色空间增强 (Tensor 类型处理)
    # 模拟显微镜亮度差异及不同染色试剂带来的饱和度变化
    def color_jitter(self, img_tensor, p=0.8):
        if np.random.random() < p:
            # 随机亮度 (Brightness)
            alpha_bright = np.random.uniform(-30, 30) / 255
            img_tensor += alpha_bright

            # 随机饱和度 (Saturation) - 针对RGB图像，通过调整通道比例模拟
            alpha_sat = np.random.uniform(0.7, 1.3)
            img_tensor[1] = img_tensor[1] * alpha_sat  # 重点对中间通道进行缩放

            img_tensor = img_tensor.clamp(0.0, 1.0)
        return img_tensor

    # 3. 高斯噪声增强 (Tensor 类型处理)
    # 模拟显微镜成像过程中的电子噪声
    def add_gaussian_noise(self, img_tensor, std=0.03, p=0.5):
        if np.random.random() < p:
            noise = torch.normal(0, std, img_tensor.shape)
            img_tensor += noise
            img_tensor = img_tensor.clamp(0.0, 1.0)
        return img_tensor


# --- 工具函数 ---


def get_image_list(image_path):
    return [f for f in os.listdir(image_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def get_label_file(label_path, image_name):
    fname = os.path.join(label_path, os.path.splitext(image_name)[0] + ".txt")
    data = []
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        with open(fname, encoding="utf-8") as f:
            for line in f:
                data.append([float(i) for i in line.strip().split()])
    return data


def save_Yolo(img, boxes, save_path, prefix, image_name):
    img_dir = os.path.join(save_path, "images")
    lbl_dir = os.path.join(save_path, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    save_img_name = prefix + image_name
    img.save(os.path.join(img_dir, save_img_name))

    save_lbl_name = prefix + os.path.splitext(image_name)[0] + ".txt"
    with open(os.path.join(lbl_dir, save_lbl_name), "w", encoding="utf-8") as f:
        for box in boxes:
            cls = int(box[0])
            coords = " ".join([f"{float(c):.6f}" for c in box[1:]])
            f.write(f"{cls} {coords}\n")


# --- 主运行函数 ---


def run_augmentation_pipeline(img_path, lbl_path, save_path, times=3):
    """Times: 每张图生成的增强变体数量."""
    aug_tool = BCCDDataAugmentation()
    image_list = get_image_list(img_path)

    print(f"开始增强处理，共找到 {len(image_list)} 张图像...")

    for img_name in image_list:
        print(f"正在处理: {img_name}")
        raw_img = Image.open(os.path.join(img_path, img_name)).convert("RGB")
        raw_boxes = torch.tensor(get_label_file(lbl_path, img_name))

        for i in range(times):
            # 1. 翻转 (克隆原始数据)
            t_img, t_boxes = aug_tool.random_flip(raw_img.copy(), raw_boxes.clone())

            # 2. 转换为 Tensor
            t_tensor = aug_tool.to_tensor(t_img)

            # 3. 颜色抖动
            t_tensor = aug_tool.color_jitter(t_tensor)

            # 4. 噪声处理
            t_tensor = aug_tool.add_gaussian_noise(t_tensor)

            # 5. 转回 PIL 并保存
            final_img = aug_tool.to_image(t_tensor)
            save_Yolo(final_img, t_boxes, save_path, prefix=f"aug{i}_", image_name=img_name)

    print("数据增强任务完成！")


if __name__ == "__main__":
    # 修改以下路径为你本地的实际路径
    IMAGE_PATH = r"E:\Deeplearning\Datasets\Br35HDet\images"
    LABEL_PATH = r"E:\Deeplearning\Datasets\Br35HDet\labels"
    SAVE_PATH = r"E:\Deeplearning\Datasets\Br35HDet\Augmentation_Final"

    run_augmentation_pipeline(IMAGE_PATH, LABEL_PATH, SAVE_PATH, times=3)
