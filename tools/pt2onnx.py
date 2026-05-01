import os

from ultralytics import YOLO

# 1. 设置路径
model_path = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\yolov8\exp2\weights\best.pt"
save_dir = os.path.dirname(model_path)
output_name = "best_optimized.onnx"
output_path = os.path.join(save_dir, output_name)

# 2. 加载模型
model = YOLO(model_path)

# 3. 执行导出与自动简化
# simplify=True 会自动调用 onnxslim 进行图优化和剪枝
print("开始导出并简化模型...")
model.export(format="onnx", opset=12, simplify=True, imgsz=640)

print(f"模型已成功导出并优化至: {output_path}")
