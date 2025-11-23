import requests
import json
import re
from typing import Dict, Any, List

class DeepSeekRAGOptimizer:
    """使用DeepSeek API的RAG优化器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"  # DeepSeek的主要模型
        
    def optimize_question(self, question: str) -> Dict[str, Any]:
        """
        使用DeepSeek API优化问题以更好地使用RAG系统
        
        Args:
            question: 原始问题字符串
            
        Returns:
            优化结果的字典
        """
        prompt = f"""
请优化以下问题以更好地在RAG系统中使用。RAG系统基于向量相似度检索，需要明确的关键词。

原始问题："{question}"

请直接以JSON格式回复，不要其他文字：

{{
    "optimized_question": "优化后的问题",
    "key_entities": ["实体1", "实体2"],
    "search_keywords": ["关键词1", "关键词2"],
    "rag_suitability_score": 1-10的评分,
    "optimization_notes": "优化说明",
    "question_type": "问题类型"
}}

优化原则：
1. 保持原意但更具体明确
2. 包含清晰的关键词便于向量检索
3. 适合从知识库文档中查找答案
4. 避免模糊、主观的表述

问题类型包括：factual(事实性), procedural(步骤性), comparative(比较性), analytical(分析性), definition(定义性)
"""
        
        try:
            response = self._call_deepseek_api(prompt)
            result = self._extract_json_from_response(response)
            result["original_question"] = question
            result["success"] = True
            result["model_used"] = self.model_name
            return result
            
        except Exception as e:
            # 如果API调用失败，返回离线优化结果
            return self._get_offline_result(question, str(e))
    
    def _call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """从响应文本中提取JSON"""
        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                raise Exception(f"JSON解析失败: {e}")
        
        # 如果提取失败，尝试清理文本后解析
        cleaned_text = text.strip()
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        raise Exception("响应中未找到有效的JSON格式")
    
    def _get_offline_result(self, question: str, error_msg: str) -> Dict[str, Any]:
        """获取离线优化结果"""
        # 简单的离线优化逻辑
        entities = self._extract_entities_offline(question)
        keywords = self._extract_keywords_offline(question)
        optimized = self._optimize_question_offline(question)
        
        return {
            "original_question": question,
            "optimized_question": optimized,
            "key_entities": entities,
            "search_keywords": keywords,
            "rag_suitability_score": self._calculate_score_offline(question, entities),
            "optimization_notes": f"DeepSeek API失败，使用离线优化: {error_msg}",
            "question_type": self._classify_question_offline(question),
            "success": False,
            "error": error_msg
        }
    
    def _extract_entities_offline(self, question: str) -> List[str]:
        """离线实体提取"""
        entities = []
        # 匹配中文实体
        chinese_entities = re.findall(r'[\u4e00-\u9fff]{2,4}', question)
        entities.extend(chinese_entities)
        # 匹配英文实体
        english_entities = re.findall(r'[A-Z][a-zA-Z0-9]+', question)
        entities.extend(english_entities)
        return list(set(entities))
    
    def _extract_keywords_offline(self, question: str) -> List[str]:
        """离线关键词提取"""
        stop_words = {'什么', '怎么', '如何', '为什么', '是否', '的', '了', '在', '是', '吗'}
        words = re.findall(r'[\u4e00-\u9fff]+', question)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords
    
    def _optimize_question_offline(self, question: str) -> str:
        """离线问题优化"""
        # 简单的优化规则
        if len(question) < 8:
            # 短问题添加上下文
            entities = self._extract_entities_offline(question)
            if entities:
                return f"详细说明：{question}"
        return question
    
    def _calculate_score_offline(self, question: str, entities: List[str]) -> int:
        """离线评分计算"""
        score = 5
        if len(entities) > 0:
            score += 2
        if 10 <= len(question) <= 50:
            score += 2
        return min(10, max(1, score))
    
    def _classify_question_offline(self, question: str) -> str:
        """离线问题分类"""
        if re.search(r'怎么|如何', question):
            return "procedural"
        elif re.search(r'是什么|定义', question):
            return "definition"
        elif re.search(r'有哪些|列举', question):
            return "list"
        elif re.search(r'比较|对比', question):
            return "comparative"
        else:
            return "factual"



# 集成到RAG系统的实用函数
def preprocess_query_with_deepseek(user_query: str, api_key: str) -> Dict[str, Any]:
    """
    使用DeepSeek预处理用户查询
    
    Args:
        user_query: 用户查询
        api_key: DeepSeek API密钥
        
    Returns:
        预处理结果
    """
    optimizer = DeepSeekRAGOptimizer(api_key)
    return optimizer.optimize_question(user_query)

def get_optimized_search_queries(user_query: str, api_key: str) -> List[str]:
    """
    获取优化后的搜索查询列表
    
    Args:
        user_query: 用户查询
        api_key: DeepSeek API密钥
        
    Returns:
        搜索查询列表
    """
    result = preprocess_query_with_deepseek(user_query, api_key)
    suggestions = [result["optimized_question"]]
    suggestions.extend(result.get("search_keywords", [])[:2])
    return suggestions

# if __name__ == "__main__":
#     # 测试DeepSeek优化器
#     API_KEY = "sk-009b68ac7a984590bf76912a64d85990"  # 请替换为您的实际API密钥
        

#     sample_query = "哆啦A梦使用的3个秘密道具分别是什么？"
#     processed = preprocess_query_with_deepseek(sample_query, API_KEY)
#     print(f"优化后查询: {processed}")
#     print(f"输入查询: {sample_query}")
#     print(f"优化后查询: {processed['optimized_question']}")
#     print(f"推荐搜索: {get_optimized_search_queries(sample_query, API_KEY)}")
#     print(f"适用性评分: {processed['rag_suitability_score']}/10")
#     print(f"处理状态: {'成功' if processed['success'] else '失败'}")