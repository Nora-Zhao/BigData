import time
import openai
import dashscope  # 通义千问SDK
from dashscope import Generation
import os

# 从环境变量读取密钥（和Shell脚本联动）
openai.api_key = os.getenv("OPENAI_API_KEY")
dashscope.api_key = os.getenv("QWEN_API_KEY")

def delayed_completion(model_company="alibaba", delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """支持通义千问和OpenAI的延迟请求函数（默认通义千问）"""
    time.sleep(delay_in_seconds)
    output, error = None, None

    for _ in range(max_trials):
        try:
            if model_company == "open_ai":
                # OpenAI调用（原有逻辑）
                output = openai.ChatCompletion.create(** kwargs)
            elif model_company == "alibaba":
                # 通义千问调用（新增逻辑）
                output = Generation.call(
                    model=kwargs.get("model", "qwen-turbo"),  # 千问模型名
                    messages=kwargs.get("messages"),          # 和OpenAI格式一致
                    result_format="message",                  # 输出格式对齐
                    max_tokens=kwargs.get("max_tokens", 10),
                    temperature=kwargs.get("temperature", 0)
                )
            else:
                raise ValueError(f"不支持的厂商：{model_company}，可选：open_ai / alibaba")
            break
        except Exception as e:
            error = e
            print(f"调用失败（重试中）：{str(e)}")  # 增加错误提示

    return output, error

def prepare_data(prompt, datum):
    """保持不变，提示词格式通用"""
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
    return prompt + input_txt + postfix

def post_process(completion, choices, model_company="alibaba"):
    """适配不同厂商的返回格式解析"""
    try:
        # 1. 根据厂商提取内容（通义千问和OpenAI返回路径不同）
        if model_company == "open_ai":
            content = completion.choices[0].message.content.strip()
        elif model_company == "alibaba":
            content = completion.output.choices[0].message.content.strip()
        else:
            raise ValueError(f"不支持的厂商：{model_company}")

        # 2. 原有逻辑：提取Choice 1/2（保持不变）
        result = []
        for choice in choices:
            choice_txt = f"Choice {choice}"  # 修复原代码少空格的问题（"Choice1"→"Choice 1"）
            if choice_txt in content:
                result.append(choice)
        return content, result
    except Exception as e:
        print(f"解析结果失败：{str(e)}")
        return "", []  # 解析失败时返回空，避免流程中断