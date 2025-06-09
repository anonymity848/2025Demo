import asyncio
import os
from openai import OpenAI


async def get_response(messages):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 填写DashScope SDK的base_url
    )

    print("model: qwen-plus")
    # 使用流模式，逐步接收生成的内容
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        stream=True
    )

    # 包装同步流对象，使其支持异步迭代
    for chunk in completion:
        # 从分片中提取出内容
        if chunk.choices:
            content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
            await asyncio.sleep(0)  # 释放控制权以允许异步操作
            yield content  # 异步生成内容分片
