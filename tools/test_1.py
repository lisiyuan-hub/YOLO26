import time
from pathlib import Path

from ultralytics import YOLO

# 1. 初始化模型
model = YOLO(
    r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\Br35HDet\runs\exp1_yolov8_CCFM_EUCB_Ghost\weights\best.pt"
)

# 2. 设置路径
input_folder = Path(r"E:\Deeplearning\yolo26\ultralytics-8.4.16\BCCD_Dataset-master\test")
output_folder = input_folder / "results"
output_folder.mkdir(exist_ok=True)

# 3. 获取图片列表
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_files = []
for ext in image_extensions:
    image_files.extend(input_folder.glob(ext))

# 4. 逐张预测并统计 FPS
total_fps = 0
num_images = len(image_files)

print(f"开始推理，共 {num_images} 张图片...")

for img_path in image_files:
    # 记录开始时间
    start_time = time.time()

    # 执行预测 (save=True 自动保存结果)
    model.predict(
        source=str(img_path),
        save=True,
        project=str(input_folder),
        name="results",
        exist_ok=True,
        verbose=False,  # 设为 False 可减少终端刷屏
    )

    # 记录结束时间
    end_time = time.time()

    # 计算耗时与 FPS
    inference_time = end_time - start_time
    fps = 1 / inference_time if inference_time > 0 else 0
    total_fps += fps

    print(f"图片: {img_path.name} | 推理耗时: {inference_time:.4f}s | FPS: {fps:.2f}")

# 5. 计算平均 FPS
avg_fps = total_fps / num_images if num_images > 0 else 0
print(f"\n检测完成！平均 FPS: {avg_fps:.2f}")
print(f"结果已保存在: {output_folder}")
