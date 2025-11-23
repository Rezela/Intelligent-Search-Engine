# 列出可用模型
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("错误: 未找到 GOOGLE_API_KEY")
else:
    genai.configure(api_key=api_key)
    print("正在列出可用模型...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"列出模型失败: {e}")