import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 注意：只需要修改一个地方：如果有多个实验对比，就在这个参数里面添加对应的实验名和results.csv的绝对路径
#  result_dict = {  'YOLO26n': r'C:\software\mydemo\yolo_demo\CSDN\YOLO26_AIguai\runs\26train\exp\results.csv',}
if __name__ == "__main__":
    # 列出待获取数据内容的文件位置
    result_dict = {
        "exp1": r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\exp1_yolov8\results.csv",
        "exp2": r"E:\Deeplearning\yolo26\ultralytics-8.4.16\runs\exp3_CCFM_SENetV2\results.csv",
    }

    # 绘制map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split(".")[-1]
        if ext == "csv":
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()  # 7是指map50的下标（每行从0开始向右数）
        else:  # 文件后缀是txt
            with open(res_path) as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[10]))
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth="1")  # 线条粗细设为1

    # 添加x轴和y轴标签
    plt.xlabel("Epochs")
    plt.ylabel("mAP@0.5")
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50.png", dpi=600)  # dpi可设为300/600/900，表示存为更高清的矢量图
    plt.show()

    # 绘制map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split(".")[-1]
        if ext == "csv":
            data = pd.read_csv(res_path, usecols=[8]).values.ravel()  # 8是指map50-95的下标（每行从0开始向右数）
        else:
            with open(res_path) as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[11]))  # 11是指map50-95的下标（每行从0开始向右数）
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth="1")

    # 添加x轴和y轴标签
    plt.xlabel("Epochs")
    plt.ylabel("mAP@0.5:0.95")
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50-95.png", dpi=600)
    plt.show()

    # 绘制训练的总loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split(".")[-1]
        if ext == "csv":
            box_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[4]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)  # 3个loss相加并且保留小数点后5位

        else:
            with open(res_path) as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[5]))
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth="1")

    # 添加x轴和y轴标签
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("loss.png", dpi=600)
    plt.show()
