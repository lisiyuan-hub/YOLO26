import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# --- 核心网络组件 (与之前一致) ---
def conv_bn_relu(in_chs, out_chs, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True),
    )


class GhostModule(nn.Module):
    def __init__(self, in_chs, out_chs, ratio=2):
        super().__init__()
        init_chs = out_chs // ratio
        new_chs = init_chs * (ratio - 1)
        self.primary_conv = conv_bn_relu(in_chs, init_chs, 3, padding=1)
        self.cheap_operation = conv_bn_relu(init_chs, new_chs, 3, padding=1)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GCE_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv_bn_relu(3, 16, 3, stride=2, padding=1)
        self.stage1 = GhostModule(16, 32)
        self.stage2 = GhostModule(32, 64)  # 对应浅层
        self.stage3 = GhostModule(64, 128)  # 对应中层
        self.stage4 = GhostModule(128, 256)  # 对应深层

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        shallow = self.stage2(s1)
        middle = self.stage3(shallow)
        deep = self.stage4(middle)
        return shallow, middle, deep


# --- 特征图保存逻辑 ---
def save_feature_map(tensor, save_path, name):
    """将 Tensor 转换为可视化图像并保存."""
    # 取通道平均值并转为 numpy
    feature_map = torch.mean(tensor, dim=1).squeeze().detach().cpu().numpy()
    # 归一化到 0-255
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)
    feature_map = (feature_map * 255).astype(np.uint8)
    # 使用伪彩色增强可视化效果（学术图中常用 JET 或 VIRIDIS）
    heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(os.path.join(save_path, f"{name}.jpg"), heatmap)
    print(f"已保存特征图: {name}.jpg 至 {save_path}")


# --- 主运行函数 ---
def run_visualization(input_img_path, output_dir):
    # 1. 加载模型
    model = GCE_Backbone().eval()

    # 2. 图像预处理 (根据 YOLO 习惯调整为 640x640)
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        input_image = Image.open(input_img_path).convert("RGB")
        input_tensor = preprocess(input_image).unsqueeze(0)
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    # 3. 推理提取特征
    with torch.no_grad():
        shallow, middle, deep = model(input_tensor)

    # 4. 自定义保存路径并输出
    save_feature_map(shallow, output_dir, "Stage_Shallow_P3")
    save_feature_map(middle, output_dir, "Stage_Middle_P4")
    save_feature_map(deep, output_dir, "Stage_Deep_P5")


# --- 用户自定义区域 ---
if __name__ == "__main__":
    # 在这里修改你的输入图片路径和输出文件夹
    USER_INPUT_PATH = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\tools\test\aug2_y699.jpg"
    USER_OUTPUT_DIR = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\tools\test\aug2_y699_output"

    run_visualization(USER_INPUT_PATH, USER_OUTPUT_DIR)
