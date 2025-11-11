import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import h5py

def load_clusters(cluster_path):
    """加载聚类结果（标签）"""
    with open(cluster_path, 'r') as f:
        clusters = json.load(f)
    return [d.get('cluster_id', -1) for d in clusters]

def load_embeddings(embed_path):
    """加载嵌入特征（用于降维）"""
    if not os.path.exists(embed_path):
        return None
    with h5py.File(embed_path, 'r') as f:
        return f['embeddings'][:]  # 适配hdf5格式的嵌入

def visualize_clusters(cluster_path, embed_path, raw_data_path, vis_dir, dataset, method="tsne"):
    # 创建保存目录
    os.makedirs(vis_dir, exist_ok=True)

    # 加载数据
    labels = load_clusters(cluster_path)
    embeddings = load_embeddings(embed_path)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"聚类数量：{n_clusters} | 样本总数：{len(labels)}")

    # 1. 降维散点图（有嵌入才绘制）
    if embeddings is not None and len(embeddings) >= len(labels):
        embeddings = embeddings[:len(labels)]  # 对齐样本数量
        # 降维
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        embeds_2d = reducer.fit_transform(embeddings)

        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Cluster Distribution of {dataset} (Dim Reduction: {method})', fontsize=12)
        plt.xlabel(f'{method} Dimension 1', fontsize=10)
        plt.ylabel(f'{method} Dimension 2', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{dataset}_cluster_scatter.png'), dpi=300)
        plt.close()
        print("✅ 散点图保存完成")

    # 2. 类别数量柱状图
    label_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Cluster Size Distribution of {dataset}', fontsize=12)
    plt.xlabel('Cluster ID', fontsize=10)
    plt.ylabel('Number of Samples', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{dataset}_cluster_sizes.png'), dpi=300)
    plt.close()
    print("✅ 柱状图保存完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_path", required=True, help="聚类结果路径（.json）")
    parser.add_argument("--embed_path", required=True, help="嵌入特征路径（.hdf5）")
    parser.add_argument("--raw_data_path", required=True, help="原始数据路径（.jsonl）")
    parser.add_argument("--vis_dir", required=True, help="可视化结果保存目录")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"], help="降维方法")
    args = parser.parse_args()
    visualize_clusters(**vars(args))
