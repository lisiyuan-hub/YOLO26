import os
import random
import shutil

def split_dataset_refined(src_images_path, src_labels_path, save_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 1. 创建目标目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(save_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_path, split, 'labels'), exist_ok=True)

    # 2. 获取所有图片
    image_extensions = ('.jpg', '.png', '.jpeg')
    all_images = [f for f in os.listdir(src_images_path) if f.lower().endswith(image_extensions)]
    
    # 3. 严格配对检查
    paired_samples = []
    for img_name in all_images:
        # 移除后缀获取文件名，寻找对应的txt
        raw_name = os.path.splitext(img_name)[0]
        label_name = raw_name + '.txt'
        
        if os.path.exists(os.path.join(src_labels_path, label_name)):
            paired_samples.append(img_name)
        else:
            print(f"跳过: {img_name} (未在 labels 文件夹找到对应 txt)")

    print(f"--- 配对统计 ---")
    print(f"图片总数: {len(all_images)}")
    print(f"成功配对总数: {len(paired_samples)}")

    if len(paired_samples) == 0:
        print("错误：未发现成对数据，请检查路径是否正确！")
        return

    # 4. 随机打乱与计算切分
    random.seed(42)
    random.shuffle(paired_samples)
    
    total = len(paired_samples)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    # 5. 分发文件
    dataset_splits = {
        'train': paired_samples[:train_end],
        'val': paired_samples[train_end:val_end],
        'test': paired_samples[val_end:]
    }

    for split_name, samples in dataset_splits.items():
        print(f"正在拷贝 {split_name} 集 (n={len(samples)})...")
        for img_name in samples:
            raw_name = os.path.splitext(img_name)[0]
            lbl_name = raw_name + '.txt'
            
            # 拷贝图片
            shutil.copy(os.path.join(src_images_path, img_name), 
                        os.path.join(save_path, split_name, 'images', img_name))
            # 拷贝标签
            shutil.copy(os.path.join(src_labels_path, lbl_name), 
                        os.path.join(save_path, split_name, 'labels', lbl_name))

    print(f"\n✅ 划分完成！数据集已保存至: {save_path}")

if __name__ == '__main__':
    # --- 请准确填写以下三个路径 ---
    SRC_IMAGES = r"E:\Deeplearning\Datasets\Br35HDet\Augmentation_Final\images"  # 增强后的图片文件夹
    SRC_LABELS = r"E:\Deeplearning\Datasets\Br35HDet\Augmentation_Final\labels"  # 增强后的标签文件夹
    SAVE_ROOT = r"E:\Deeplearning\Datasets\Br35HDet\Br35HDet_Final_Split_1"                  # 最终保存的根目录

    split_dataset_refined(SRC_IMAGES, SRC_LABELS, SAVE_ROOT)