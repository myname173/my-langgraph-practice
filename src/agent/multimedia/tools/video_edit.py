# src/agent/multimedia/tools/video_edit.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def edit_video_style(video_url: str, prompt: str) -> str:
    """调用 wan2.7-videoedit 进行视频后期精修（已适配 2026 年最新 media 结构）"""
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    session = requests.Session()
    session.trust_env = False
    
    # ✅ 正确 endpoint（视频编辑模型专用）
    submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    
    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # ✅ 关键修复：使用官方要求的 media 数组结构
    payload = {
        "model": "wan2.7-videoedit",
        "input": {
            "prompt": prompt,                    # 后期精修提示词
            "media": [                           # ← 必填！新版必须使用 media
                {
                    "type": "video",             # 视频编辑固定使用 "video"
                    "url": video_url
                }
            ]
        },
        "parameters": {
            "resolution": "1080P",               # 可选：720P / 1080P（1080P 画质更好）
            "prompt_extend": True,               # 开启 prompt 智能改写，提升后期效果
            "watermark": False                   # 不加水印
        }
    }
    
    response = session.post(submit_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"后期精修提交异常: {response.text}")
    
    task_id = response.json().get("output", {}).get("task_id")
    print(f"[后期进度] 正在进行电影级调色与特效增强，Task ID: {task_id}")
    
    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    while True:
        poll_resp = session.get(poll_url, headers=headers)
        poll_data = poll_resp.json()
        output = poll_data.get("output", {})
        task_status = output.get("task_status", "")
        
        if task_status == "SUCCEEDED":
            # 更鲁棒的 URL 解析（兼容新旧返回结构）
            final_video_url = (
                output.get("video_url") or
                output.get("url") or
                (output.get("results", [{}])[0].get("url") if output.get("results") else None)
            )
            if final_video_url:
                print(f"    ✅ 后期精修完成！URL: {final_video_url}")
                return final_video_url
            raise Exception(f"后期精修成功但未能解析视频 URL: {poll_data}")
        
        elif task_status == "FAILED":
            raise Exception(f"后期精修失败: {poll_data}")
        
        print("    [后期进度] 视频逐帧渲染精修中，请耐心等待 (约5秒/次)...")
        time.sleep(5)
