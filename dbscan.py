import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# --------------------------
# 1. 读取并预处理日志数据
# --------------------------
def preprocess_log(line):
    """清洗日志文本，移除噪声"""
    # 移除时间戳（如：2023-10-01 12:34:56）
    line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', line)
    # 移除IP地址（如：192.168.1.1）
    line = re.sub(r'\d+\.\d+\.\d+\.\d+', '', line)
    # 移除数字和特殊符号
    line = re.sub(r'[0-9\W]', ' ', line)
    # 移除过长单词（可能是随机ID或噪声）
    line = ' '.join([word for word in line.split() if 2 <= len(word) <= 15])
    return line.strip()

# 读取日志文件（大文件逐行处理，过滤空行）
log_file = 'BGL-500MB/BGL_500M.log'
log_lines = []
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        processed = preprocess_log(line)
        if processed:  # 只保留非空行
            log_lines.append(processed)

# 转为DataFrame
data = pd.DataFrame(log_lines, columns=['cleaned_message'])
print(f"处理后的数据量：{len(data)} 条日志")

# --------------------------
# 2. 特征提取（TF-IDF）
# --------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,  # 限制最大特征数，避免维度爆炸
    ngram_range=(1, 2)  # 同时考虑单词和双词组合
)
X = vectorizer.fit_transform(data['cleaned_message'])
print(f"TF-IDF特征维度：{X.shape}")

# --------------------------
# 3. 数据标准化与降维
# --------------------------
# 标准化（稀疏矩阵关闭均值中心化）
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# 降维（用TruncatedSVD替代PCA，支持稀疏矩阵）
svd = TruncatedSVD(n_components=50, random_state=42)  # 先降到50维用于聚类
X_svd = svd.fit_transform(X_scaled)
print(f"降维后维度：{X_svd.shape}")

# 为了可视化，再降到2维
svd_vis = TruncatedSVD(n_components=2, random_state=42)
X_vis = svd_vis.fit_transform(X_scaled)

# --------------------------
# 4. DBSCAN聚类与参数调优
# --------------------------
# 小规模数据测试最优参数（样本量过大时建议采样）
sample_size = min(10000, len(X_svd))  # 最多采样10000条
sample_indices = pd.Series(range(len(X_svd))).sample(sample_size, random_state=42).values
X_sample = X_svd[sample_indices]

# 网格搜索最优参数
best_score = -1
best_params = {'eps': 0.5, 'min_samples': 5}  # 默认参数

for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for min_samples in [5, 10, 15, 20]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X_sample)
        # 至少有2个有效簇且噪声比例不超过50%才计算分数
        if len(set(labels)) > 1 and (labels == -1).mean() < 0.5:
            score = silhouette_score(X_sample, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"最优参数：{best_params}，轮廓系数：{best_score:.3f}")

# 用最优参数全量聚类
dbscan = DBSCAN(**best_params, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_svd)

# 统计聚类结果
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"聚类完成：{n_clusters} 个簇，噪声点：{n_noise} 个（{n_noise/len(dbscan_labels):.2%}）")

# --------------------------
# 5. 可视化聚类结果并保存（不弹出窗口）
# --------------------------
plt.figure(figsize=(12, 8))
# 噪声点用灰色，簇用彩色
scatter = plt.scatter(
    X_vis[:, 0], X_vis[:, 1],
    c=dbscan_labels,
    cmap='tab10',
    alpha=0.6,
    s=5,
    edgecolors='none'
)
plt.title(f'DBSCAN Clustering (eps={best_params["eps"]}, min_samples={best_params["min_samples"]})', fontsize=14)
plt.xlabel('SVD Component 1', fontsize=12)
plt.ylabel('SVD Component 2', fontsize=12)
plt.colorbar(scatter, label='Cluster Label (-1 = Noise)')
plt.tight_layout()
# 保存图片（300dpi高清，避免标签截断）
plt.savefig('bgl_log_dbscan_clustering.png', dpi=300, bbox_inches='tight')
print("DBSCAN聚类可视化图已保存为：bgl_log_dbscan_clustering.png")

# --------------------------
# 6. 保存聚类结果（日志+簇标签+是否噪声）到CSV
# --------------------------
data['cluster_label'] = dbscan_labels
data['is_noise'] = (data['cluster_label'] == -1).astype(int)  # 新增噪声标记列
data.to_csv('bgl_log_dbscan_results.csv', index=False, encoding='utf-8')
print("聚类结果（日志+簇标签+噪声标记）已保存为：bgl_log_dbscan_results.csv")

# --------------------------
# 7. 输出每个簇的代表性日志（控制台打印）
# --------------------------
for cluster_id in set(dbscan_labels):
    if cluster_id == -1:
        print(f"\n噪声点代表性日志（前5条）：")
        noise_samples = data[data['cluster'] == -1]['cleaned_message'].head(5)
        for i, sample in enumerate(noise_samples, 1):
            print(f"{i}. {sample}")
    else:
        print(f"\n簇 {cluster_id} 的代表性日志（前5条）：")
        cluster_samples = data[data['cluster'] == cluster_id]['cleaned_message'].head(5)
        for i, sample in enumerate(cluster_samples, 1):
            print(f"{i}. {sample}")