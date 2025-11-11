import time
import os
import openai
import dashscope
from dashscope import Generation

# 通义千问：环境变量名 QWEN_API_KEY
dashscope.api_key = os.getenv("QWEN_API_KEY")
# OpenAI：环境变量名 OPENAI_API_KEY（保留原有逻辑）
openai.api_key = os.getenv("OPENAI_API_KEY")

def delayed_completion(model_company="alibaba", delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """默认使用通义千问，支持环境变量读取API Key"""
    # 先检查对应厂商的密钥是否存在
    if model_company == "alibaba" and not dashscope.api_key:
        raise ValueError("请配置环境变量 QWEN_API_KEY（通义千问密钥）")
    if model_company == "open_ai" and not openai.api_key:
        raise ValueError("请配置环境变量 OPENAI_API_KEY（OpenAI密钥）")

    time.sleep(delay_in_seconds)
    output, error = None, None

    for _ in range(max_trials):
        try:
            if model_company == "open_ai":
                output = openai.ChatCompletion.create(**kwargs)
            elif model_company == "alibaba":
                output = Generation.call(
                    model=kwargs.get("model", "qwen-turbo"),
                    messages=kwargs.get("messages"),
                    result_format="message",
                    max_tokens=kwargs.get("max_tokens", 1024),
                    temperature=kwargs.get("temperature", 0.7),
                )
            else:
                raise ValueError(f"不支持的厂商：{model_company}，可选：open_ai / alibaba")
            break
        except Exception as e:
            error = e
            print(f"调用失败（重试中）：{str(e)}")

    return output, error

def post_process(completion, model_company="alibaba"):
    """默认按通义千问解析"""
    try:
        if model_company == "open_ai":
            content = completion.choices[0].message.content.strip()
        elif model_company == "alibaba":
            content = completion.output.choices[0].message.content.strip()
        else:
            raise ValueError(f"不支持的厂商：{model_company}")

        result = []
        if 'Yes' in content and 'No' not in content:
            result.append('Yes')
        elif 'No' in content and 'Yes' not in content:
            result.append('No')
        return content, result
    except Exception as e:
        print(f"解析失败：{str(e)}")
        return "", []