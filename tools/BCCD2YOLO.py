import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(xml_path, img_dir, output_dir, class_mapping):
    """单文件转换核心函数"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 1. 获取图像尺寸
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in class_mapping:
                continue
            cls_id = class_mapping[cls_name]

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 2. 归一化处理
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # 防止浮点误差越界
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 3. 写入txt文件
        txt_filename = os.path.splitext(os.path.basename(xml_path))[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    except Exception as e:
        print(f"处理 {xml_path} 时出错: {e}")

# --- 批量处理入口 ---
if __name__ == "__main__":
    # 【配置区域】请根据你的实际路径修改
    XML_DIR = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\BCCD_Dataset-master\BCCD\Annotations"   # XML存放路径
    IMG_DIR = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\BCCD_Dataset-master\BCCD\JPEGImages"        # 图片存放路径
    OUT_DIR = r"E:\Deeplearning\yolo26\ultralytics-8.4.16\BCCD_Dataset-master\BCCD\labels"        # 转换后TXT存放路径
    
    # 【类别映射】必须对应血细胞数据集XML里的标签名称
    # 请打开一个XML确认名称，例如有的叫 'RBC'，有的叫 'red blood cell'
    MAPPING = {
        "WBC": 0,
        "RBC": 1,
        "Platelets": 2
    }

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # 循环处理所有文件
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"开始转换，共找到 {len(xml_files)} 个标注文件...")
    
    for xml_name in xml_files:
        xml_full_path = os.path.join(XML_DIR, xml_name)
        convert_voc_to_yolo(xml_full_path, IMG_DIR, OUT_DIR, MAPPING)
        
    print("转换完成！TXT标签已保存到 labels 文件夹。")