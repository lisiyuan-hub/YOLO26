import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据集加载与探索
# ==========================================
# 加载 Sklearn 内置的手写数字数据集 (8x8 像素)
digits = load_digits()
X = digits.data    # 特征矩阵: (1797, 64)
y = digits.target  # 目标标签: 0-9
target_names = digits.target_names

print(f"成功加载数据集，样本数: {X.shape[0]}, 原始特征维度: {X.shape[1]}")

# 展示部分原始数字图像
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {y[i]}", fontsize=8)
    plt.axis('off')
plt.suptitle("Original 8x8 Grayscale Images")
plt.show()

# ==========================================
# 2. 数据标准化 (重要预处理)
# ==========================================
# PCA 对特征缩放敏感，先进行标准化使均值为0，方差为1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. PCA 维度压缩 (64维 -> 2维)
# ==========================================
# 实例化 PCA 模型，设定目标维度为 2
pca = PCA(n_components=2)

# 执行拟合和转换
X_pca = pca.fit_transform(X_scaled)

# 计算解释方差贡献率
variance_ratio = pca.explained_variance_ratio_
print(f"\n主成分分析完成:")
print(f"第一主成分 (PC1) 解释方差比: {variance_ratio[0]:.4f}")
print(f"第二主成分 (PC2) 解释方差比: {variance_ratio[1]:.4f}")
print(f"总累计解释方差: {sum(variance_ratio):.4f}")

# ==========================================
# 4. 输出压缩后的样本集数据
# ==========================================
# 将降维后的数据转为 DataFrame 方便查看
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = y

print("\n压缩后的样本集（前10条数据预览）：")
print(pca_df.head(10).to_string(index=False))

# ==========================================
# 5. 结果可视化分析
# ==========================================
plt.figure(figsize=(10, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i, color in zip(range(10), colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, alpha=0.7, label=str(i), s=30)

plt.legend(title="Digit Labels", loc='best', ncol=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Visualization of Digits Dataset after PCA (64D to 2D)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()