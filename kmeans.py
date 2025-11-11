import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  # 替代PCA，支持稀疏矩阵
import matplotlib.pyplot as plt

# --------------------------
# 1. 读取日志数据
# --------------------------
# 逐行读取日志，过滤空行（避免无效数据）
with open('BGL-500MB/BGL_500M.log', 'r', encoding='utf-8', errors='ignore') as file:
    log_lines = [line.strip() for line in file if line.strip()]

data = pd.DataFrame(log_lines, columns=['log_message'])
print(f"成功读取 {len(data)} 条有效日志")

# --------------------------
# 2. TF-IDF特征提取
# --------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000  # 保留Top 10000个高频词，避免维度爆炸
)
X = vectorizer.fit_transform(data['log_message'])  # 稀疏矩阵
print(f"TF-IDF特征维度：{X.shape}")

# --------------------------
# 3. 标准化与降维
# --------------------------
scaler = StandardScaler(with_mean=False)  # 稀疏矩阵关闭均值中心化
X_scaled = scaler.fit_transform(X)

svd = TruncatedSVD(n_components=2, random_state=42)  # 降到2维用于可视化
X_svd = svd.fit_transform(X_scaled)
print(f"降维后维度：{X_svd.shape}")

# --------------------------
# 4. K-means聚类
# --------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_svd)
print(f"聚类完成，共分为 {len(set(kmeans_labels))} 个簇")

# --------------------------
# 5. 绘图并保存（不弹出窗口，直接存文件）
# --------------------------
plt.figure(figsize=(10, 6))  # 设置图片尺寸
plt.scatter(
    X_svd[:, 0], X_svd[:, 1],
    c=kmeans_labels,
    cmap='tab10',  # 更清晰的颜色映射
    alpha=0.6,     # 调整透明度，避免点重叠
    s=8            # 点大小，适配大规模数据
)
plt.title('K-means Clustering of BGL Log Messages', fontsize=14)
plt.xlabel('SVD Component 1', fontsize=12)
plt.ylabel('SVD Component 2', fontsize=12)
plt.colorbar(label='Cluster Label')  # 添加颜色条，标注簇编号
plt.tight_layout()  # 自动调整布局，避免标签截断

# 保存图片（支持png/jpg/pdf等格式，dpi越高越清晰）
plt.savefig('bgl_log_kmeans_clustering.png', dpi=300, bbox_inches='tight')
print("聚类可视化图已保存为：bgl_log_kmeans_clustering.png")

# 可选：保存聚类结果（日志文本 + 簇标签）到CSV文件
data['cluster_label'] = kmeans_labels
data.to_csv('bgl_log_clustering_results.csv', index=False, encoding='utf-8')
print("聚类结果（日志+簇标签）已保存为：bgl_log_clustering_results.csv")