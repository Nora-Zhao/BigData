#!/bin/bash
set -e  # 遇到错误立即退出

# ==================== 核心配置参数（已适配BGL_500M_J2.jsonl）====================
OPENAI_API_KEY="sk-cad4c76c92bd4ed3b52149954285bf0c"  # 有效OpenAI密钥（仅用GPT-3.5）
OPENAI_API_URL="https://api.openai.com/v1"  # 需代理则替换为代理地址
DATASET="bgl_log"                      # 数据集名称（自定义，与目录一致）
SCALE="BGL_500M_J2"                    # 数据文件名（无需后缀，需与datasets目录文件匹配）
NUM_TRIPLETS=10000                     # 采样三元组数量（根据数据量调整）
MODEL_NAME="hkunlp/instructor-large"   # 固定嵌入模型（CLUSTERLLM依赖）
BATCH_SIZE=32                          # 微调批次大小（CPU建议改为8）
EPOCHS=5                               # 微调轮数
LEARNING_RATE=2e-5                     # 微调学习率
CUDA_DEVICE="0"                        # 有GPU填"0/1"，无GPU填"none"
CLUSTERLLM_DIR="ClusterLLM"            # 仓库根目录（当前脚本在BigData目录下，无需修改）
OPENAI_ORG=""                          # 无组织ID则留空

# ==================== 自动推导绝对路径（避免cd后路径失效）====================
SCRIPT_DIR=$(cd $(dirname $0); pwd)  # 获取脚本所在绝对路径（BigData目录）
CLUSTERLLM_ABS_DIR="${SCRIPT_DIR}/${CLUSTERLLM_DIR}"  # ClusterLLM绝对路径

# 所有路径改为绝对路径
RAW_DATA_PATH="${CLUSTERLLM_ABS_DIR}/datasets/${SCALE}.jsonl"
EMBED_RAW_PATH="${CLUSTERLLM_ABS_DIR}/datasets/${SCALE}_embeds.hdf5"
TRIPLET_SAMPLE_PATH="${CLUSTERLLM_ABS_DIR}/perspective/1_predict_triplet/sampled_triplet_results/${DATASET}/${SCALE}_triplets.jsonl"
TRIPLET_PRED_PATH="${CLUSTERLLM_ABS_DIR}/perspective/1_predict_triplet/predicted_triplet_results/${DATASET}/${SCALE}_predicted_triplets.jsonl"
CHECKPOINT_DIR="${CLUSTERLLM_ABS_DIR}/perspective/2_finetune/checkpoints/${DATASET}"
EMBED_FINETUNE_PATH="${CHECKPOINT_DIR}/${SCALE}_embeds.hdf5"
CLUSTER_RESULT_PATH="${CHECKPOINT_DIR}/${SCALE}_clusters.json"
VIS_DIR="${CLUSTERLLM_ABS_DIR}/visualization_results_p"
VIS_SCRIPT="${CLUSTERLLM_ABS_DIR}/visualize_clusters_p.py"

# ==================== 前置检查与环境准备 ====================
# 1. 检查原始数据是否存在
if [ ! -f "$RAW_DATA_PATH" ]; then
    echo "❌ 原始数据文件不存在：$RAW_DATA_PATH"
    echo "请确认 datasets 目录下有文件：${SCALE}.jsonl"
    exit 1
fi

# 2. 检查API密钥是否有效
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY 未填写，无法调用GPT-3.5-turbo"
    exit 1
fi

# 3. 创建所有输出目录
mkdir -p $(dirname "$TRIPLET_SAMPLE_PATH")
mkdir -p $(dirname "$TRIPLET_PRED_PATH")
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$VIS_DIR"

## 4. 安装缺失依赖（自动补全核心库）
#echo -e "\n=== 检查并安装依赖包 ==="
#pip install -q openai dashscope instructor transformers torch scikit-learn pandas h5py matplotlib numpy

# ==================== 步骤1：生成原始嵌入 ====================
echo -e "\n=== 步骤1：生成原始嵌入 ==="
cd "${CLUSTERLLM_ABS_DIR}/perspective/2_finetune"  # 绝对路径切换，避免失效
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 get_embedding.py \
        --model_name "$MODEL_NAME" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$EMBED_RAW_PATH" \
        --measure
else
    python3 get_embedding.py \
        --model_name "$MODEL_NAME" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$EMBED_RAW_PATH" \
        --measure
fi
echo "✅ 原始嵌入生成完成（路径：$EMBED_RAW_PATH）"

# ==================== 步骤2：采样并预测三元组 ====================
echo -e "\n=== 步骤2：采样并预测三元组 ==="
cd "${CLUSTERLLM_ABS_DIR}/perspective/1_predict_triplet"  # 绝对路径切换

# 采样三元组
python3 sample_triplet.py \
    --dataset "$DATASET" \
    --scale "$SCALE" \
    --data_path "$RAW_DATA_PATH" \
    --output_dir "$(dirname $TRIPLET_SAMPLE_PATH)" \
    --num_triplets "$NUM_TRIPLETS"
echo "✅ 三元组采样完成（路径：$TRIPLET_SAMPLE_PATH）"

# LLM预测三元组（传递API环境变量）
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_API_BASE="$OPENAI_API_URL"
export OPENAI_ORG="$OPENAI_ORG"
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 predict_triplet.py \
        --dataset "$DATASET" \
        --scale "$SCALE" \
        --triplet_path "$TRIPLET_SAMPLE_PATH" \
        --output_dir "$(dirname $TRIPLET_PRED_PATH)" \
        --model "gpt-3.5-turbo" \
        --api_base "$OPENAI_API_URL"
else
    python3 predict_triplet.py \
        --dataset "$DATASET" \
        --scale "$SCALE" \
        --triplet_path "$TRIPLET_SAMPLE_PATH" \
        --output_dir "$(dirname $TRIPLET_PRED_PATH)" \
        --model "gpt-3.5-turbo" \
        --api_base "$OPENAI_API_URL"
fi
echo "✅ LLM预测三元组完成（路径：$TRIPLET_PRED_PATH）"

# ==================== 步骤3：转换三元组格式并微调模型 ====================
echo -e "\n=== 步骤3：转换三元组并微调模型 ==="
cd "${CLUSTERLLM_ABS_DIR}/perspective/2_finetune"  # 绝对路径切换

# 转换三元组格式
python3 convert_triplet.py \
    --triplet_path "$TRIPLET_PRED_PATH" \
    --output_dir "${CLUSTERLLM_ABS_DIR}/perspective/2_finetune/converted_triplet_results/${DATASET}" \
    --split train
echo "✅ 三元组格式转换完成"

# 微调模型（GPU/CPU适配）
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 finetune.py \
        --model_name "$MODEL_NAME" \
        --train_triplet_path "${CLUSTERLLM_ABS_DIR}/perspective/2_finetune/converted_triplet_results/${DATASET}/train_triplets.jsonl" \
        --output_dir "$CHECKPOINT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE"
else
    python3 finetune.py \
        --model_name "$MODEL_NAME" \
        --train_triplet_path "${CLUSTERLLM_ABS_DIR}/perspective/2_finetune/converted_triplet_results/${DATASET}/train_triplets.jsonl" \
        --output_dir "$CHECKPOINT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE"
fi
echo "✅ 模型微调完成（路径：$CHECKPOINT_DIR）"

# ==================== 步骤4：生成微调后嵌入并聚类 ====================
echo -e "\n=== 步骤4：生成微调后嵌入并聚类 ==="
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 get_embedding.py \
        --model_name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT_DIR" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$EMBED_FINETUNE_PATH" \
        --measure \
        --overwrite
else
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 get_embedding.py \
        --model_name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT_DIR" \
        --scale "$SCALE" \
        --task_name "$DATASET" \
        --data_path "$RAW_DATA_PATH" \
        --result_file "$EMBED_FINETUNE_PATH" \
        --measure \
        --overwrite
fi
echo "✅ 微调后嵌入及聚类完成（嵌入：$EMBED_FINETUNE_PATH；聚类：$CLUSTER_RESULT_PATH）"

# ==================== 步骤5：聚类结果可视化 ====================
echo -e "\n=== 步骤5：聚类结果可视化 ==="
cd "${CLUSTERLLM_ABS_DIR}"  # 回到仓库根目录
if [ "$CUDA_DEVICE" != "none" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 "$VIS_SCRIPT" \
        --cluster_path "$CLUSTER_RESULT_PATH" \
        --embed_path "$EMBED_FINETUNE_PATH" \
        --raw_data_path "$RAW_DATA_PATH" \
        --vis_dir "$VIS_DIR" \
        --dataset "$DATASET" \
        --method "tsne"
else
    python3 "$VIS_SCRIPT" \
        --cluster_path "$CLUSTER_RESULT_PATH" \
        --embed_path "$EMBED_FINETUNE_PATH" \
        --raw_data_path "$RAW_DATA_PATH" \
        --vis_dir "$VIS_DIR" \
        --dataset "$DATASET" \
        --method "tsne"
fi
echo "✅ 可视化完成（结果路径：$VIS_DIR）"

# ==================== 流程结束 ====================
echo -e "\n🎉 所有核心流程执行完成！"
echo "📁 关键输出文件汇总："
echo "1. 微调模型：$CHECKPOINT_DIR"
echo "2. 微调后嵌入：$EMBED_FINETUNE_PATH"
echo "3. 聚类结果：$CLUSTER_RESULT_PATH"
echo "4. 可视化图表：$VIS_DIR（含TSNE散点图+聚类分布柱状图）"