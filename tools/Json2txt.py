import json
import os


def convert_via_to_yolo(json_path, output_dir, img_w=640, img_h=640):
    """针对 Br35HDet VIA 格式的转换器 支持 Polygon 和 Ellipse 形状自动转为 YOLO 矩形框.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path) as f:
        data = json.load(f)

    for key, value in data.items():
        file_name = value["filename"]
        regions = value.get("regions", [])

        if not regions:
            continue

        # 准备对应的 txt 文件
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(output_dir, txt_name), "w") as f_out:
            for region in regions:
                shape = region["shape_attributes"]
                shape_type = shape.get("name")

                # --- 情况 1: 处理多边形 (Polygon) ---
                if shape_type == "polygon":
                    x_pts = shape["all_points_x"]
                    y_pts = shape["all_points_y"]
                    xmin, xmax = min(x_pts), max(x_pts)
                    ymin, ymax = min(y_pts), max(y_pts)

                # --- 情况 2: 处理椭圆 (Ellipse) ---
                elif shape_type == "ellipse":
                    cx, cy = shape["cx"], shape["cy"]
                    rx, ry = shape["rx"], shape["ry"]
                    # 简化处理：取其水平轴和垂直轴作为边界
                    xmin, xmax = cx - rx, cx + rx
                    ymin, ymax = cy - ry, cy + ry

                # --- 情况 3: 处理圆形 (Circle) ---
                elif shape_type == "circle":
                    cx, cy, r = shape["cx"], shape["cy"], shape["r"]
                    xmin, xmax = cx - r, cx + r
                    ymin, ymax = cy - r, cy + r

                else:
                    continue  # 忽略未知形状

                # 计算 YOLO 格式所需的归一化中心点和宽高
                w = xmax - xmin
                h = ymax - ymin
                x_center = (xmin + w / 2) / img_w
                y_center = (ymin + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # 写入文件 (类别 ID 默认为 0，代表 Tumor)
                f_out.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"转换成功！共生成 {len(data)} 个标签文件。")


# --- 使用设置 ---
# 请注意：img_w 和 img_h 必须设置为你图片的真实像素大小（Br35HDet 通常是 640x640）
convert_via_to_yolo(
    r"E:\Deeplearning\Datasets\Br35HDet\Br35HDet\annotations_val.json",
    r"E:\Deeplearning\Datasets\Br35HDet\Br35HDet\val\labels",
    img_w=640,
    img_h=640,
)
