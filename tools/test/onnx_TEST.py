import time

import cv2
import numpy as np
import onnxruntime as ort

# 1. 加载模型
onnx_path = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\BCCD\exp1_CCFM_Ghost_EUCB_DetectStrong\weights\best.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def predict_onnx(image_path, iterations=50):
    # 准备图片
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640)).transpose((2, 0, 1))
    input_tensor = img_resized[np.newaxis, ...].astype(np.float32) / 255.0
    input_name = session.get_inputs()[0].name

    # --- 预热 (Warmup) ---
    print("正在预热模型...")
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # --- 开始计时 ---
    print(f"开始测试 {iterations} 次平均推理速度...")
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: input_tensor})
    end_time = time.time()

    # --- 计算指标 ---
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    fps = 1000 / avg_time_ms

    print("-" * 30)
    print(f"平均推理耗时: {avg_time_ms:.2f} ms")
    print(f"推理 FPS: {fps:.2f}")
    print("-" * 30)


# 执行测试
predict_onnx(r"E:\Deeplearning\yolo26\ultralytics-8.4.16\BCCD_Dataset-master\test\aug2_y547.jpg")
