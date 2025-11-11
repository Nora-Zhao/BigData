import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def load_data(pred_path, raw_data_path, feat_path):
    # 加载聚类预测结果（含每个样本的类别）
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    # 加载原始数据（可选，用于文本信息）
    with open(raw_data_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    # 加载嵌入特征（用于降维，无则返回None）
    feats = None
    if feat_path and os.path.exists(feat_path):
        import h5py
        with h5py.File(feat_path, 'r') as f:
            feats = f['embeddings'][:]  # 假设嵌入向量存在'hdf5'的'embeddings'键下
    return pred_data, raw_data, feats

def visualize_clusters(pred_path, raw_data_path, feat_path, vis_dir, dataset, method="tsne"):
    # 创建保存目录
    os.makedirs(vis_dir, exist_ok=True)
    pred_data, raw_data, feats = load_data(pred_path, raw_data_path, feat_path)

    # 1. 提取聚类标签（假设每个样本的类别在'prediction'字段中）
    labels = [d.get('prediction', -1) for d in pred_data]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"聚类数量：{n_clusters}")

    # 2. 降维可视化（若有嵌入特征）
    if feats is not None and len(feats) > 0:
        # 降维（TSNE或PCA）
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        feats_2d = reducer.fit_transform(feats[:len(labels)])  # 对齐样本数量

        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Cluster Label')
        plt.title(f'Cluster Distribution of {dataset} (降维方法：{method})')
        plt.xlabel(f'{method} Dimension 1')
        plt.ylabel(f'{method} Dimension 2')
        plt.savefig(os.path.join(vis_dir, f'{dataset}_cluster_scatter.png'), dpi=300)
        plt.close()

    # 3. 类别数量柱状图
    label_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar')
    plt.title(f'Cluster Size Distribution of {dataset}')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{dataset}_cluster_sizes.png'), dpi=300)
    plt.close()
    print(f"可视化结果已保存至：{vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="聚类预测结果路径")
    parser.add_argument("--raw_data_path", required=True, help="原始数据路径")
    parser.add_argument("--feat_path", default=None, help="嵌入特征路径（用于降维）")
    parser.add_argument("--vis_dir", required=True, help="可视化结果保存目录")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"], help="降维方法")
    args = parser.parse_args()
    visualize_clusters(**vars(args))
