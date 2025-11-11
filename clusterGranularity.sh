#!/bin/bash
set -e  # 遇到错误立即退出

# ==================== 核心配置（已适配国内网络，无需科学上网）====================
DATASET="BGL_500M_J1"                  # 数据集名称（需与JSONL文件名一致）
SCALE="small"                          # 采样规模（固定）
EMBED_METHOD="finetuned"               # 嵌入方法（固定）
D="67.0"                               # 聚类参数（固定）
OPENAI_API_KEY="sk-cad4c76c92bd4ed3b52149954285bf0c"  # 你的OpenAI密钥
OPENAI_ORG=""                          # 无组织ID则留空
SEED=100                               # 随机种子
CUDA_DEVICE="0"                        # 有GPU填"0"，无GPU填"none"
EMBED_MODEL="all-MiniLM-L6-v2"         # 改用国内可下载的轻量模型（无需科学上网）

# ==================== 自动推导路径（无需修改）====================
BASE_DIR=$(cd $(dirname $0); pwd)
CLUSTERLLM_DIR="${BASE_DIR}/ClusterLLM"

# 文件路径
RAW_DATA_PATH="${CLUSTERLLM_DIR}/datasets/${DATASET}.jsonl"
EMBED_OUTPUT_DIR="${CLUSTERLLM_DIR}/perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/${EMBED_MODEL}-${DATASET}-d=${D}-epoch=15/checkpoint-3840"
FEAT_PATH="${EMBED_OUTPUT_DIR}/${SCALE}_embeds.hdf5"
OUT_DIR="${CLUSTERLLM_DIR}/sampled_pair_results"
PROMPT_PATH="${CLUSTERLLM_DIR}/prompts_pair_exps_pair_v8.json"
CLUSTERING_RESULTS="${OUT_DIR}/${DATASET}_embed=${EMBED_METHOD}_s=${SCALE}_k=1_multigran2-200_seed=${SEED}.json"
PRED_PAIR_DIR="${CLUSTERLLM_DIR}/predicted_pair_results"
PRED_PAIR_PATH="${PRED_PAIR_DIR}/${DATASET}_embed=${EMBED_METHOD}_s=${SCALE}_k=1_multigran2-200_seed=${SEED}-gpt-4-0314-prompts_pair_exps_pair_v3.json"
VIS_DIR="${CLUSTERLLM_DIR}/visualization_results_g"
VIS_SCRIPT="${CLUSTERLLM_DIR}/visualize_clusters_g.py"

# ==================== 前置检查 ====================
# 1. 检查原始数据
if [ ! -f "$RAW_DATA_PATH" ]; then
    echo "❌ 原始数据文件不存在：$RAW_DATA_PATH"
    echo "请确保 ${CLUSTERLLM_DIR}/datasets/ 目录下有 ${DATASET}.jsonl 文件"
    exit 1
fi

# 2. 检查OpenAI密钥
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 未设置OPENAI_API_KEY，请填写有效密钥"
    exit 1
fi

# 3. 创建输出目录
mkdir -p "$EMBED_OUTPUT_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$PRED_PAIR_DIR"
mkdir -p "$VIS_DIR"

## 4. 安装依赖（确保sentence-transformers版本兼容）
#echo -e "\n=== 安装/更新依赖包 ==="
#pip install -q h5py openai pandas matplotlib scikit-learn sentence-transformers==2.2.2

# ==================== 步骤0：生成嵌入文件（改用国内可下载模型）====================
echo -e "\n=== 步骤0：生成嵌入文件（.hdf5）==="
# 关键修改：用 all-MiniLM-L6-v2（轻量、国内可直接下载，无需科学上网）
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 "${CLUSTERLLM_DIR}/perspective/2_finetune/get_embedding.py" \
        --model_name "$EMBED_MODEL" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$FEAT_PATH" \
        --measure
else
    python3 "${CLUSTERLLM_DIR}/perspective/2_finetune/get_embedding.py" \
        --model_name "$EMBED_MODEL" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$FEAT_PATH" \
        --measure
fi

# 验证嵌入文件
if [ ! -f "$FEAT_PATH" ]; then
    echo "❌ 嵌入文件生成失败，请检查网络或 get_embedding.py 脚本"
    exit 1
fi
echo "✅ 嵌入文件生成成功（路径：$FEAT_PATH）"

# ==================== 步骤1：采样样本对 ====================
echo -e "\n=== 步骤1：采样样本对 ==="
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 "${CLUSTERLLM_DIR}/granularity/sample_pairs.py" \
        --dataset "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --feat_path "$FEAT_PATH" \
        --scale "$SCALE" \
        --embed_method "$EMBED_METHOD" \
        --k 1 \
        --out_dir "$OUT_DIR" \
        --min_clusters 2 \
        --max_clusters 200 \
        --seed "$SEED"
else
    python3 "${CLUSTERLLM_DIR}/granularity/sample_pairs.py" \
        --dataset "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --feat_path "$FEAT_PATH" \
        --scale "$SCALE" \
        --embed_method "$EMBED_METHOD" \
        --k 1 \
        --out_dir "$OUT_DIR" \
        --min_clusters 2 \
        --max_clusters 200 \
        --seed "$SEED"
fi

if [ ! -f "$CLUSTERING_RESULTS" ]; then
    echo "❌ 样本对采样失败，请检查 sample_pairs.py 脚本"
    exit 1
fi
echo "✅ 样本对采样完成（路径：$CLUSTERING_RESULTS）"

# ==================== 步骤2：生成提示词文件 ====================
echo -e "\n=== 步骤2：生成提示词文件 ==="
python3 "${CLUSTERLLM_DIR}/granularity/sample_pairs_for_prompt.py" \
    --prompt_path "$PROMPT_PATH" \
    --sampled_pair_path "$CLUSTERING_RESULTS" \
    --data_path "$RAW_DATA_PATH" \
    --dataset "$DATASET" \
    --seed 1234

echo "✅ 提示词文件生成完成（路径：$PROMPT_PATH）"

# ==================== 步骤3：GPT-4预测样本对 ====================
echo -e "\n=== 步骤3：GPT-4预测样本对 ==="
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_ORG="$OPENAI_ORG"

if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 "${CLUSTERLLM_DIR}/granularity/predict_pairs.py" \
        --dataset "$DATASET" \
        --data_path "$CLUSTERING_RESULTS" \
        --model_name "gpt-4-0314" \
        --openai_org "$OPENAI_ORG" \
        --prompt_file "$PROMPT_PATH" \
        --temperature 0 \
        --output_dir "$PRED_PAIR_DIR"
else
    python3 "${CLUSTERLLM_DIR}/granularity/predict_pairs.py" \
        --dataset "$DATASET" \
        --data_path "$CLUSTERING_RESULTS" \
        --model_name "gpt-4-0314" \
        --openai_org "$OPENAI_ORG" \
        --prompt_file "$PROMPT_PATH" \
        --temperature 0 \
        --output_dir "$PRED_PAIR_DIR"
fi

echo "✅ GPT-4预测完成（结果路径：$PRED_PAIR_DIR）"

# ==================== 步骤4：预测聚类数量 ====================
echo -e "\n=== 步骤4：预测聚类数量 ==="
python3 "${CLUSTERLLM_DIR}/granularity/predict_num_clusters.py" \
    --dataset "$DATASET" \
    --embed_method "$EMBED_METHOD" \
    --clustering_results "$CLUSTERING_RESULTS" \
    --pred_path "$PRED_PAIR_PATH" \
    --min_clusters 2 \
    --max_clusters 200

echo "✅ 聚类数量预测完成"

# ==================== 步骤5：聚类结果可视化 ====================
echo -e "\n=== 步骤5：聚类结果可视化 ==="
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 "$VIS_SCRIPT" \
        --pred_path "$PRED_PAIR_PATH" \
        --raw_data_path "$RAW_DATA_PATH" \
        --feat_path "$FEAT_PATH" \
        --vis_dir "$VIS_DIR" \
        --dataset "$DATASET" \
        --method "tsne"
else
    python3 "$VIS_SCRIPT" \
        --pred_path "$PRED_PAIR_PATH" \
        --raw_data_path "$RAW_DATA_PATH" \
        --feat_path "$FEAT_PATH" \
        --vis_dir "$VIS_DIR" \
        --dataset "$DATASET" \
        --method "tsne"
fi

echo "✅ 可视化完成（结果路径：$VIS_DIR）"

# ==================== 完成 ====================
echo -e "\n🎉 全流程执行完成！"
echo "关键结果路径："
echo "1. 嵌入文件：$FEAT_PATH"
echo "2. 采样样本对：$CLUSTERING_RESULTS"
echo "3. GPT预测结果：$PRED_PAIR_DIR"
echo "4. 可视化图表：$VIS_DIR"