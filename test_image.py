#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图片分析功能的脚本
"""

import base64
import io
from SourceAPI import HANDLERS
import PIL.Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_image_analysis():
    """测试图片分析功能"""

    # 创建一个简单的测试图片 (红色方块)
    img = PIL.Image.new('RGB', (100, 100), color='red')

    # 保存到内存并编码为base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()

    # 编码为base64 (模拟前端上传格式)
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    image_data = f"data:image/png;base64,{base64_data}"

    # 获取图片分析处理器
    handler = HANDLERS.get("image_analysis_api")
    if not handler:
        print("❌ 找不到 image_analysis_api 处理器")
        return

    # 测试通用描述
    print("🧪 测试图片通用描述...")
    result1 = handler.handle("", {"image_data": image_data})
    print(f"结果: {result1}")

    # 测试具体问题
    print("\n🧪 测试图片问答...")
    result2 = handler.handle("这张图片是什么颜色的？", {"image_data": image_data})
    print(f"结果: {result2}")

    # 测试错误情况 - 无图片数据
    print("\n🧪 测试错误情况 - 无图片数据...")
    result3 = handler.handle("测试", {})
    print(f"结果: {result3}")

if __name__ == "__main__":
    test_image_analysis()
