from ultralytics import YOLO
import torch

def train_custom_model():
    # 1. 初始化模型：加载你修改后的 YAML 结构
    # 这会根据 YAML 创建一个新的模型实例，参数是随机初始化的
    model = YOLO(r"E:\Deeplearning\yolo26\ultralytics-8.4.16\ultralytics\cfg\models\26\yolo26_EVA.yaml") 

    # 2. 加载预训练权重 (.pt)
    # 核心技巧：使用 .load() 而不是在 YOLO() 中直接传入路径
    # 框架会自动匹配相同名称和形状的 Tensor，忽略 EVA 模块等不匹配部分
    weights_path = 'yolo26n.pt' 
    model.load(weights_path)
    
    print(f"成功从 {weights_path} 加载了兼容层的权重。")

    # 3. 冻结部分层（可选，建议初期冻结 Backbone 以提升初速）
    # 假设前 10 层是 Backbone，你可以通过以下方式冻结
    freeze_layers = 10 
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False

    # 4. 开始训练
    model.train(
        data=r"E:\Deeplearning\yolo26\ultralytics-8.4.16\ultralytics\cfg\datasets\VisDrone.yaml",      # 你的数据集配置文件
        epochs=300,            # 训练轮数
        imgsz=640,             # 输入图像尺寸
        batch=8,              # 批处理大小
        lr0=0.001,              # 初始学习率 (如果冻结了建议设小点，如 0.001)
        optimizer='AdamW',     # 推荐使用 AdamW 来稳定大核注意力模块 [cite: 183]
        device=0,              # 使用 GPU 0
        project='YOLO26_EVA',  # 项目名称
        name='exp1_eva_added',  # 实验名称
        cache=True,            # 是否缓存数据
        workers=0,             # 数据加载工作进程数
    )

if __name__ == '__main__':
    train_custom_model()