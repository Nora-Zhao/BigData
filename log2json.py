import re
import json
import os

# 日志路径（根据你的实际路径修改）
LOG_PATH = "BGL-500MB/BGL_500M.log"
OUTPUT_DIR = "BGL-500MB"
OUTPUT_JSONL = f"{OUTPUT_DIR}/BGL_500M_J.jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_bgl_log_v2(line):
    """精准清洗：只删噪声，保留核心信息（先提取首字段判断标签，再清洗日志内容）"""
    # 先分割首字段（判断正常/异常的依据）和日志主体
    parts = line.strip().split(maxsplit=1)  # 只分割首字段和后续内容
    if len(parts) < 2:
        return "", ""  # 无效日志，返回空

    first_field = parts[0]
    log_body = parts[1]

    # 判断标签：首字段为'-' → normal；否则 → abnormal（英文标签）
    label = "normal" if first_field == "-" else "abnormal"

    # 清洗日志主体（保留核心信息，删除噪声）
    # 1. 移除时间戳（2005-06-14-09.53.46.860321、2005.06.14）
    log_body = re.sub(r'\d{4}[-./]\d{2}[-./]\d{2}(-\d{2}\.\d{2}\.\d{2}\.\d+)?', '', log_body)
    # 2. 移除节点ID（R01-M1-N1-C:J15-U01 这类格式）
    log_body = re.sub(r'[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+:[A-Z0-9]+-[A-Z0-9]+', '', log_body)
    # 3. 移除数字串（纯数字ID，如 1118768026、0x00004ed8 保留，因为是错误地址）
    log_body = re.sub(r'\b\d{8,12}\b', '', log_body)
    # 4. 移除多余空格和连字符（开头/结尾的）
    log_body = re.sub(r'^[- ]+|[- ]+$', '', log_body)
    # 5. 合并多个空格
    log_body = re.sub(r'\s+', ' ', log_body)

    return label, log_body.strip()


# 处理日志（保留核心信息，可选采样）
count = 0
normal_count = 0
abnormal_count = 0
max_count = 100000  # 可选：限制最大条数，避免数据过大
with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f_in, \
        open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        label, cleaned_body = clean_bgl_log_v2(line)
        # 保留有效日志（内容长度>5，避免空文本）
        if cleaned_body and len(cleaned_body) > 5:
            # 输出格式：input（清洗后日志）+ label（normal/abnormal），适配脚本
            json.dump({
                "input": cleaned_body,
                "label": label
            }, f_out, ensure_ascii=False)
            f_out.write('\n')
            count += 1
            if label == "normal":
                normal_count += 1
            else:
                abnormal_count += 1


print(f"Processing completed! Total valid logs: {count}")
print(f"Normal logs: {normal_count}, Abnormal logs: {abnormal_count}")
print("\nSample results (label + input):")
# 打印前3条验证
with open(OUTPUT_JSONL, 'r') as f:
    for i in range(3):
        line = f.readline()
        if line:
            data = json.loads(line)
            print(f"- label: {data['label']}, input: {data['input']}")