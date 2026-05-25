import os
import shutil


def organize_dataset(source_dir, target_dir):
    # 定义目标路径
    img_dir = os.path.join(target_dir, "images")
    lbl_dir = os.path.join(target_dir, "labels")

    # 如果文件夹不存在则创建
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 支持的图片格式
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

    # 遍历源文件夹
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)

        # 跳过目录，只处理文件
        if os.path.isfile(source_path):
            if file_name.lower().endswith(img_extensions):
                shutil.move(source_path, os.path.join(img_dir, file_name))
                print(f"移动图片: {file_name}")
            elif file_name.lower().endswith(".txt"):
                shutil.move(source_path, os.path.join(lbl_dir, file_name))
                print(f"移动标签: {file_name}")


# --- 使用设置 ---
# source: 存放混杂文件的文件夹
# target: 想要存放的根目录
source = r"E:\Deeplearning\Datasets\RCS-YOLO-main\RCS-YOLO-main\dataset-Br35H\valdata"
target = r"E:\Deeplearning\Datasets\RCS-YOLO-main\RCS-YOLO-main\dataset-Br35H\t"

organize_dataset(source, target)
