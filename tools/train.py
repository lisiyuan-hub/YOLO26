import sys
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

# 自动定位项目根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def train_model(data_yaml, model_path_input, epochs, imgsz, batch, device):
    # 路径处理优化
    model_path = Path(model_path_input)
    cfg_base = Path(r"E:\Deeplearning\yolo26\ultralytics-8.4.16\ultralytics\cfg")
    data_path = cfg_base / "datasets" / data_yaml
    project_dir = Path(r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\yolov8")
    
    print(f"DEBUG: 正在加载模型文件: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"文件不存在: {model_path}")
        
    model = YOLO(str(model_path))
    
    # 开始训练：严格匹配描述中的实验设置
    model.train(
        # --- 基础配置 ---
        data=str(data_path),
        epochs=epochs,         # 200轮
        imgsz=imgsz,          # 640
        batch=batch,          # 16
        device=device,
        project=str(project_dir),
        name='YOLO_GCE_exp',
        exist_ok=False,
        
        optimizer='AdamW',     # 明确使用 AdamW
        lr0=1e-3,             # 初始学习率 10⁻³
        lrf=0.01,             # 最终学习率倍率（余弦退火，通常设为0.01）
        weight_decay=5e-4,     # 权重衰减 5×10⁻⁴
        warmup_epochs=5.0,     # 前 5 轮线性预热
        cos_lr=True,          # 开启余弦退火调度
        
        mosaic=1.0,           # 开启 Mosaic 增强
        flipud=0.5,           # 随机翻转（上下）
        fliplr=0.5,           # 随机翻转（左右）
        degrees=10.0,         # 随机旋转度数
        # close_mosaic=10,    # 可选：最后10轮关闭Mosaic以稳定收敛
        
        workers=4,            # 4060 性能尚可，建议 4-8
        cache=True,           # 开启缓存提速
        amp=True,             # 4060 支持混合精度
        deterministic=True    # 为了学术实验可复现性，建议设为 True
    )

def parse_opt():
    parser = argparse.ArgumentParser()
    # 对应描述：BCCD 或 Br35HDet
    parser.add_argument('--data', type=str, default="BCCD.yaml") 
    # 指向你的自定义 YOLO-GCE 模型结构文件
    parser.add_argument('--model', type=str, default=r"E:\Deeplearning\yolo26\ultralytics-8.4.16\ultralytics\cfg\models\v8\yolov8_CCFM_Ghost_EUCB.yaml")
    parser.add_argument('--epochs', type=int, default=200)   # 对应描述 200 轮
    parser.add_argument('--imgsz', type=int, default=640)    # 对应描述 640x640
    parser.add_argument('--batch', type=int, default=16)     # 对应描述 16
    parser.add_argument('--device', type=str, default='0')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    train_model(opt.data, opt.model, opt.epochs, opt.imgsz, opt.batch, opt.device)