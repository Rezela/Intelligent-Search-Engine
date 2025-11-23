import os
import json
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 GOOGLE_API_KEY，请检查 .env 文件")

        # 配置 Google API
        genai.configure(api_key=self.api_key)

        # 自动选择最佳可用模型
        self.model_name = self._get_best_available_model()
        logging.info(f"GeminiClient 已初始化，使用模型: {self.model_name}")

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def _get_best_available_model(self):
        preferred_order = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]

        try:
            all_models = genai.list_models()
            available_models = [m.name.replace("models/", "") for m in all_models if
                                'generateContent' in m.supported_generation_methods]

            for preference in preferred_order:
                if preference in available_models:
                    return preference

            if available_models:
                return available_models[0]

        except Exception as e:
            logging.error(f"无法列出模型列表: {e}")

        return "gemini-pro"

    def chat(self, system_prompt, user_prompt, max_tokens=1000, temperature=0.7):
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt
            )

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            response = model.generate_content(
                user_prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )

            # --- 修复的部分开始 ---
            content = ""
            finish_reason = "UNKNOWN"

            # 方法 A: 使用 SDK 提供的便捷属性 (推荐)
            try:
                content = response.text
                return {"content": content, "raw": str(response)}
            except Exception:
                # 如果因为安全原因被拦截，response.text 会抛出异常，此时我们手动解析
                pass

            # 方法 B: 手动解析 Candidate 结构
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name

                # 正确的结构是 candidate.content.parts
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text
                else:
                    logging.warning(f"Gemini 返回内容为空，结束原因: {finish_reason}")
                    return {
                        "content": "抱歉，由于安全策略，内容无法显示。",
                        "error": f"Finish Reason: {finish_reason}",
                        "raw": str(response)
                    }
            # --- 修复的部分结束 ---

            return {
                "content": content,
                "raw": str(response)
            }

        except Exception as e:
            logging.error(f"Gemini API 调用失败: {e}")
            return {
                "content": "",
                "error": str(e)
            }


HKGAIClient = GeminiClient

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("-" * 30)
        client = GeminiClient()
        print(f"✅ 成功连接! 当前选中的模型是: {client.model_name}")
        print("正在发送测试消息...")

        sys_p = "你是一个幽默的助手。"
        user_p = "简单介绍一下你自己 (Gemini)。"

        result = client.chat(sys_p, user_p)

        print("-" * 30)
        if result.get("content"):
            print(f"回答:\n{result['content']}")
        else:
            print(f"❌ 测试失败: {result.get('error')}")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")