# src/agent/multimedia/tools/image_gen.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_keyframe(prompt: str, size: str = "2K", reference_image_url: str = None) -> str:
    """调用通义万相 wan2.7-image-pro 生成关键帧（支持参考图）"""
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    session = requests.Session()
    session.trust_env = False
    
    submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
    
    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 组装 content，如果传入了参考图，则放入 content 列表
    content_list =[]
    if reference_image_url:
        content_list.append({"image": reference_image_url})
    content_list.append({"text": prompt})
    
    payload = {
        "model": "wan2.7-image",
        "input": {
            "messages":[
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        },
        "parameters": {
            "size": size,
            "n": 1,
            "watermark": False
        }
    }
    
    response = session.post(submit_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"画师提交异常: {response.text}")
    
    task_id = response.json().get("output", {}).get("task_id")
    print(f"[画师进度] 关键帧渲染中，Task ID: {task_id}")
    
    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    while True:
        poll_resp = session.get(poll_url, headers=headers)
        poll_data = poll_resp.json()
        output = poll_data.get("output", {})
        task_status = output.get("task_status", "")
        
        if task_status == "SUCCEEDED":
            url = None
            if "choices" in output and output["choices"]:
                content = output["choices"][0].get("message", {}).get("content",[])
                if content and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            url = item.get("image") or item.get("url")
                            if url:
                                break
            
            if not url and "url" in output:
                url = output["url"]
            
            if not url:
                results = output.get("results",[])
                if results and isinstance(results, list):
                    url = results[0].get("image") or results[0].get("url")
            
            if url and isinstance(url, str) and url.startswith("http"):
                print(f"    ✅ 关键帧生成成功！URL: {url}")
                return url
            else:
                raise Exception(f"画师出图成功但未能解析 URL: {poll_data}")
        
        elif task_status == "FAILED":
            raise Exception(f"画师出图失败: {poll_data}")
        
        time.sleep(3)
