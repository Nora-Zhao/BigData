import os
import argparse
import json
import openai
import dashscope  # 新增：通义千问SDK
from tqdm import tqdm
from tools import delayed_completion, prepare_data, post_process  # 假设delayed_completion已支持多厂商


def predict(args):
    # 1. 初始化模型密钥（区分厂商）
    if args.model_company == "open_ai":
        # OpenAI配置（保留原有逻辑）
        openai.organization = args.openai_org  # 仅OpenAI需要
        openai.api_key = os.getenv("OPENAI_API_KEY")
    elif args.model_company == "alibaba":
        # 通义千问配置（无需organization）
        dashscope.api_key = os.getenv("QWEN_API_KEY")  # 从环境变量读密钥
    else:
        raise ValueError(f"不支持的厂商：{args.model_company}")

    # 2. 生成预测结果保存路径（保持不变）
    pred_path = args.data_path.split("/")[-1].replace(
        ".json",
        f"-{args.model_name}{'-temp' + str(round(args.temperature, 1)) if args.temperature > 0 else ''}-pred.json"
    )
    pred_path = os.path.join("predicted_triplet_results", pred_path)
    print("预测结果保存路径: ", pred_path)

    # 3. 加载数据（保持不变）
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)

    # 4. 加载提示词（保持不变）
    with open("prompts.json", 'r') as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    # 5. 预处理数据（保持不变）
    for d in data:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)

    # 6. 调用模型预测（核心修改：加入model_company）
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if idx == 0:
            print("提示词示例: ", datum['prepared'])  # 打印第一个示例调试
        if 'prediction' in datum:  # 跳过已预测的数据
            continue

        # 构造请求消息（格式和OpenAI一致，通义千问兼容）
        messages = [{"role": "user", "content": datum['prepared']}]

        # 调用改造后的delayed_completion，指定厂商
        completion, error = delayed_completion(
            model_company=args.model_company,  # 新增：指定用通义千问或OpenAI
            delay_in_seconds=args.delay,
            max_trials=args.max_trials,
            model=args.model_name,  # 通义千问模型名如"qwen-turbo"
            messages=messages,
            max_tokens=10,
            temperature=args.temperature
        )

        # 处理预测结果（错误处理保持不变）
        if completion is None:
            print(f"第{idx + 1}条预测失败，保存当前结果")
            with open(pred_path, 'w') as f:
                json.dump(data, f)
            print("错误信息: ", error)
            break
        else:
            # 调用post_process时传入厂商，适配返回格式
            content, results = post_process(
                completion,
                datum['options'],
                model_company=args.model_company  # 新增：告诉post_process用哪种格式解析
            )
            data[idx]['content'] = content
            data[idx]['prediction'] = results

        # 定期保存结果（保持不变）
        if idx % args.save_every == 0 and idx > 0:
            print(f"第{idx + 1}条预测完成，保存结果")
            with open(pred_path, "w") as f:
                json.dump(data, f)

    # 最终保存（保持不变）
    with open(pred_path, "w") as f:
        json.dump(data, f)
    print("所有预测完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 核心新增：模型厂商参数（默认alibaba）
    parser.add_argument("--model_company", type=str, default="alibaba", choices=["alibaba", "open_ai"],
                        help="模型厂商：alibaba（通义千问）/ open_ai（OpenAI）")
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, required=True)
    # 调整：openai_org改为可选（仅OpenAI需要）
    parser.add_argument("--openai_org", type=str, default="",
                        help="OpenAI组织ID（仅当model_company=open_ai时有效）")
    parser.add_argument("--model_name", type=str, default="qwen-turbo",  # 默认通义千问模型
                        help="模型名称（通义千问：qwen-turbo/qwen-plus；OpenAI：gpt-3.5-turbo等）")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)  # 通义千问默认0更稳定
    args = parser.parse_args()

    predict(args)