import re
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --------------------------
# 1. 读取并预处理日志（减少噪声和数据量）
# --------------------------
def clean_log(line):
    """清洗日志：移除时间戳、IP、数字等噪声"""
    line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', line)  # 时间戳
    line = re.sub(r'\d+\.\d+\.\d+\.\d+', '', line)  # IP地址
    line = re.sub(r'[0-9\W]', ' ', line)  # 数字和特殊符号
    line = ' '.join([w for w in line.split() if 2 <= len(w) <= 15])  # 过滤过短/过长单词
    return line.strip()

# 读取日志（大文件逐行处理，过滤空行）
log_file = 'BGL-500MB/BGL_500M.log'
log_lines = []
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        cleaned = clean_log(line)
        if cleaned:
            log_lines.append(cleaned)

# 若数据量过大，采样减少样本数（层次聚类对大规模数据不友好）
sample_size = min(5000, len(log_lines))  # 最多保留5000个样本（根据内存调整）
if len(log_lines) > sample_size:
    random.seed(42)
    log_lines = random.sample(log_lines, sample_size)

data = pd.DataFrame(log_lines, columns=['cleaned_message'])
print(f"处理后样本数：{len(data)}")

# --------------------------
# 2. 特征提取（TF-IDF）
# --------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,  # 限制特征数，减少维度
    ngram_range=(1, 2)  # 包含单词和双词组合
)
X = vectorizer.fit_transform(data['cleaned_message'])  # 稀疏矩阵
print(f"TF-IDF特征维度：{X.shape}")

# --------------------------
# 3. 标准化与降维
# --------------------------
# 标准化（稀疏矩阵关闭均值中心化）
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# 降维（用TruncatedSVD替代PCA，支持稀疏矩阵）
svd = TruncatedSVD(n_components=50, random_state=42)  # 先降到50维保留信息
X_svd = svd.fit_transform(X_scaled)

# 再降到2维用于可视化
svd_vis = TruncatedSVD(n_components=2, random_state=42)
X_vis = svd_vis.fit_transform(X_scaled)

# --------------------------
# 4. 层次聚类（优化参数和效率）
# --------------------------
# 层次聚类（使用降维后的50维数据）
Z = linkage(X_svd, method='ward', metric='euclidean')

# 生成聚类标签（指定聚类数量为3）
n_clusters = 3
hierarchical_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
print(f"聚类完成，共分为 {n_clusters} 个簇")

# 绘制谱系图并保存（样本数<1000时生成，否则跳过避免过密）
if len(X_svd) <= 1000:
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=50, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=12)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
    print("谱系图已保存为：hierarchical_dendrogram.png")
else:
    print("样本数>1000，跳过谱系图绘制（避免图形过密）")

# --------------------------
# 5. 聚类结果可视化并保存
# --------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_vis[:, 0], X_vis[:, 1],
    c=hierarchical_labels,
    cmap='tab10',
    alpha=0.7,
    s=10
)
plt.title(f'Hierarchical Clustering (n_clusters={n_clusters})', fontsize=14)
plt.xlabel('SVD Component 1', fontsize=12)
plt.ylabel('SVD Component 2', fontsize=12)
plt.colorbar(scatter, label='Cluster Label')
plt.tight_layout()
plt.savefig('bgl_log_hierarchical_clustering.png', dpi=300, bbox_inches='tight')
print("聚类可视化图已保存为：bgl_log_hierarchical_clustering.png")

# --------------------------
# 6. 保存聚类结果（日志+簇标签）到CSV
# --------------------------
data['cluster_label'] = hierarchical_labels
data.to_csv('bgl_log_hierarchical_results.csv', index=False, encoding='utf-8')
print("聚类结果（日志+簇标签）已保存为：bgl_log_hierarchical_results.csv")

# --------------------------
# 7. 输出每个簇的代表性日志（控制台打印）
# --------------------------
for cluster_id in range(1, n_clusters + 1):
    print(f"\n簇 {cluster_id} 的代表性日志（前5条）：")
    cluster_samples = data[data['cluster'] == cluster_id]['cleaned_message'].head(5)
    for i, sample in enumerate(cluster_samples, 1):
        print(f"{i}. {sample}")