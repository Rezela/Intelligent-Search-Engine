import os
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from dotenv import load_dotenv
from typing import List, Union, Dict, Any, Optional
import PIL.Image

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
        
        # 宽松的安全设置，防止误杀
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def _get_best_available_model(self) -> str:
        """根据环境自动选择最佳模型"""
        preferred_order = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",
            "gemini-2.0-flash-thinking-exp",
            # "gemini-3-pro-preview",
            # "gemini-3-pro-image-preview",
            # "nano-banana-pro-preview",
        ]
        try:
            all_models = genai.list_models()
            available_models = [m.name.replace("models/", "") for m in all_models if 'generateContent' in m.supported_generation_methods]
            
            for preference in preferred_order:
                if preference in available_models:
                    return preference
            
            if available_models:
                return available_models[0]
        except Exception as e:
            logging.error(f"无法列出模型列表: {e}")
        return "gemini-pro"

    def _init_model(self, system_instruction: Optional[str] = None):
        """内部方法：初始化模型实例"""
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )

    def generate_content(self, contents: Union[str, List[Any]], system_instruction: str = None, max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """
        [多模态] 通用生成接口 (单轮)
        支持纯文本、图片、或 文本+图片 混合输入
        :param contents: 可以是字符串，或者包含 [str, PIL.Image, ...] 的列表
        :param max_tokens: 最大输出token数
        :param temperature: 温度参数 (0.0-1.0)
        """
        try:
            model = self._init_model(system_instruction)
            
            # 构建 generation_config
            generation_config = None
            if max_tokens is not None or temperature is not None:
                generation_config = genai.types.GenerationConfig()
                if max_tokens is not None:
                    generation_config.max_output_tokens = max_tokens
                if temperature is not None:
                    generation_config.temperature = temperature
            
            # 调用 API
            kwargs = {
                "contents": contents,
                "safety_settings": self.safety_settings
            }
            if generation_config:
                kwargs["generation_config"] = generation_config
                
            response = model.generate_content(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            logging.error(f"Generate Content 失败: {e}")
            return {"content": "", "error": str(e)}

    def chat_with_history(self, message: Union[str, List[Any]], history: List[Dict[str, Any]] = [], system_instruction: str = None, max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """
        [多轮对话] 聊天接口
        :param message: 当前用户的新消息 (文本或多模态)
        :param history: 历史消息列表，格式需符合 Gemini 标准 [{'role': 'user'|'model', 'parts': [...]}]
        :param max_tokens: 最大输出token数
        :param temperature: 温度参数 (0.0-1.0)
        """
        try:
            model = self._init_model(system_instruction)
            chat_session = model.start_chat(history=history)
            
            # 构建 generation_config
            kwargs = {
                "message": message,
                "safety_settings": self.safety_settings
            }
            if max_tokens is not None or temperature is not None:
                generation_config = genai.types.GenerationConfig()
                if max_tokens is not None:
                    generation_config.max_output_tokens = max_tokens
                if temperature is not None:
                    generation_config.temperature = temperature
                kwargs["generation_config"] = generation_config
                
            response = chat_session.send_message(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            logging.error(f"Chat Session 失败: {e}")
            return {"content": "", "error": str(e)}

    # 这里的参数定义必须明确包含 max_tokens 和 temperature
    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """
        [兼容旧接口] 简单的单轮文本对话
        保留此方法以兼容 RAG.py 等旧代码
        :param max_tokens: 最大输出token数
        :param temperature: 温度参数 (0.0-1.0)
        """
        return self.generate_content(
            contents=user_prompt, 
            system_instruction=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def _parse_response(self, response) -> Dict[str, Any]:
        """统一解析 Gemini 响应"""
        content = ""
        finish_reason = "UNKNOWN"
        try:
            # 尝试直接获取文本 (SDK 智能处理)
            content = response.text
        except Exception:
            # 处理安全拦截或其他异常情况
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text
                else:
                    return {
                        "content": "内容被安全策略拦截或无法生成。",
                        "error": f"Finish Reason: {finish_reason}",
                        "raw": str(response)
                    }
        
        return {"content": content, "raw": str(response)}

    @staticmethod
    def format_history(app_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        辅助工具：将简单的应用历史格式转换为 Gemini API 格式
        假设 app_history 格式: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        gemini_history = []
        for msg in app_history:
            role = 'user' if msg.get('role') == 'user' else 'model'
            content = msg.get('content', '')
            if content:
                gemini_history.append({'role': role, 'parts': [content]})
        return gemini_history

# 兼容性别名
HKGAIClient = GeminiClient

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = GeminiClient()
    print(f"✅ 模型就绪: {client.model_name}")

    # --- 测试 1: 纯文本 (单轮) ---
    print("\n--- 测试 1: 文本生成 ---")
    res = client.generate_content("用emoji画一只猫")
    print(f"回答: {res['content']}")

    # --- 测试 2: 多轮对话 ---
    print("\n--- 测试 2: 多轮对话 ---")
    history = [
        {"role": "user", "parts": ["你好，我叫小明"]},
        {"role": "model", "parts": ["你好小明！很高兴认识你。"]}
    ]
    res_chat = client.chat_with_history("我刚才说了我叫什么？", history=history)
    print(f"回答: {res_chat['content']}")

    # --- 测试 3: 多模态 (如果本地有图片) ---
    # try:
    #     img = PIL.Image.open("test_image.jpg")
    #     print("\n--- 测试 3: 图片理解 ---")
    #     res_img = client.generate_content(["这张图里有什么？", img])
    #     print(f"回答: {res_img['content']}")
    # except Exception:
    #     print("\n(跳过图片测试: 未找到 test_image.jpg)")