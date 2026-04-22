# src/agent/multimedia/tools/text_llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import httpx

load_dotenv()

http_client = httpx.Client(trust_env=False)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    http_client=http_client
)

def call_llm(prompt_text: str, role_name: str = "LLM") -> str:
    """通用的 LLM 调用接口"""
    try:
        # 对总导演增加温度，其它角色保持较低温度
        temperature = 0.9 if "总导演" in role_name else 0.7
        
        response = client.chat.completions.create(
            model="tongyi-xiaomi-analysis-pro", 
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,      # ← 新增：提高创意
            top_p=0.95,                   # ← 新增：增加多样性
            presence_penalty=0.2,       # 可选，减少重复词
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"{role_name} 思考失败: {str(e)}")
