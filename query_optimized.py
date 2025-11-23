import google.generativeai as genai
import json
import os
from typing import Dict, Any
import re

class RAGQueryOptimizer:
    def __init__(self, api_key: str):
        """初始化优化器"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """从响应文本中提取JSON"""
        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 如果提取失败，返回默认结构
        return {
            "optimized_question": text.strip(),
            "key_entities": [],
            "search_keywords": [],
            "rag_suitability_score": 5,
            "optimization_notes": "自动提取失败，使用原始问题"
        }
    
    def optimize_for_rag(self, question: str) -> Dict[str, Any]:
        """
        优化问题以更好地使用RAG系统
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
    "optimization_notes": "优化说明"
}}

优化原则：
1. 保持原意但更具体
2. 包含明确的关键词
3. 适合向量检索
4. 避免模糊表述
"""
        
        try:
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            result["original_question"] = question
            return result
            
        except Exception as e:
            # 如果API调用失败，返回基础优化
            return {
                "original_question": question,
                "optimized_question": question,
                "key_entities": self._extract_entities_simple(question),
                "search_keywords": self._extract_keywords_simple(question),
                "rag_suitability_score": 5,
                "optimization_notes": f"API调用失败: {str(e)}",
                "error": True
            }
    
    def _extract_entities_simple(self, text: str) -> list:
        """简单实体提取"""
        entities = []
        # 提取可能的名词短语
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        for word in words:
            if len(word) > 1:  # 过滤单字
                entities.append(word)
        return entities
    
    def _extract_keywords_simple(self, text: str) -> list:
        """简单关键词提取"""
        # 移除标点和常见虚词
        stop_words = {'什么', '怎么', '如何', '为什么', '是否', '的', '了', '在', '是'}
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

def test_optimization():
    """测试优化功能"""
    API_KEY = "AIzaSyBcvYd9nH6oXPRYsdXgS7-jghY7cLDdjas"  # 请替换为您的API密钥
    
    try:
        optimizer = RAGQueryOptimizer(API_KEY)
        
        test_questions = [
            "哆啦A梦使用的3个秘密道具分别是什么？",
            "北京今天的天气情况",
            "科大到中环要多久",
            "中国石化今天的收盘价是多少",
            "今天关于OpenAI的最新新闻",
            "如何学习Python编程？需要哪些步骤？"
        ]
        
        print("开始RAG优化分析...")
        print("=" * 80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 60)
            
            result = optimizer.optimize_for_rag(question)
            
            print(f"原始问题: {result['original_question']}")
            print(f"优化后问题: {result['optimized_question']}")
            print(f"关键实体: {result.get('key_entities', [])}")
            print(f"搜索关键词: {result.get('search_keywords', [])}")
            print(f"RAG适用性评分: {result.get('rag_suitability_score', 'N/A')}")
            print(f"优化说明: {result.get('optimization_notes', 'N/A')}")
            
            if result.get('error'):
                print("⚠️  使用备用方案优化")
            
            print("=" * 80)
            
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查API密钥是否正确")

# 简化但更稳定的版本
def simple_rag_optimize(question: str, api_key: str) -> Dict[str, Any]:
    """
    简化的RAG问题优化函数
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""优化这个问题用于RAG检索: "{question}"
        
输出JSON格式:
{{"optimized": "优化后问题", "keywords": ["词1","词2"]}}"""
        
        response = model.generate_content(prompt)
        
        # 简单提取JSON
        text = response.text.strip()
        if text.startswith('{') and text.endswith('}'):
            result = json.loads(text)
        else:
            # 如果没有正确JSON，使用原始问题
            result = {"optimized": question, "keywords": question.replace('?','').replace('？','').split()}
        
        return {
            "original": question,
            "optimized": result.get("optimized", question),
            "keywords": result.get("keywords", []),
            "success": True
        }
        
    except Exception as e:
        # 完全失败时的备用方案
        return {
            "original": question,
            "optimized": question,
            "keywords": re.findall(r'[\u4e00-\u9fff]{2,}', question),
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # 方法1: 使用类版本
    test_optimization()
    
    # 方法2: 使用简化版本
    print("\n" + "="*80)
    print("简化版本测试:")
    print("="*80)
    
    API_KEY = "AIzaSyBcvYd9nH6oXPRYsdXgS7-jghY7cLDdjas"
    test_questions = ["哆啦A梦使用的3个秘密道具分别是什么？", "北京今天的天气情况"]
    
    for q in test_questions:
        result = simple_rag_optimize(q, API_KEY)
        print(f"问题: {q}")
        print(f"优化: {result['optimized']}")
        print(f"关键词: {result['keywords']}")
        print(f"成功: {result['success']}")
        print("-" * 40)
