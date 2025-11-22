import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from api import HKGAIClient


INTENT_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "weather_api",
        "description": "用户询问实时天气、气温、空气质量或气象趋势，通常包含地名。",
        "required_entities": ["location"],
        "example_queries": [
            "北京今天的天气怎么样？",
            "上海明天下雨吗？"
        ],
    },
    {
        "name": "traffic_api",
        "description": "咨询从 A 地到 B 地的交通时间、路线、距离等出行问题，常出现“到、去、多久”等词。",
        "required_entities": ["origin", "destination"],
        "example_queries": [
            "科大到中环要多久？",
            "从深圳去广州开车多远？"
        ],
    },
    {
        "name": "finance_api",
        "description": "关注股票/基金/汇率等金融行情，例如收盘价、涨跌幅。",
        "required_entities": ["ticker"],
        "example_queries": [
            "中国石化今天收盘价是多少？",
            "比特币最新行情"
        ],
    },
    {
        "name": "rag",
        "description": "上述类别都不匹配时，默认交给本地知识库 RAG 管线处理。",
        "required_entities": [],
        "example_queries": [
            "哆啦A梦使用的三个秘密道具是什么？"
        ],
    },
]

SCHEMA_DESCRIPTION = json.dumps(
    {
        "intent": "string，必须是 weather_api / traffic_api / finance_api / rag 之一",
        "confidence": "float，取值 0-1，代表模型对该 intent 的置信度",
        "reason": "string，20 字以内解释判定依据",
        "entities": {
            "location": "string or null，天气问题识别到的地名",
            "origin": "string or null，交通起点",
            "destination": "string or null，交通终点",
            "ticker": "string or null，金融问题中的股票或资产代码/名称",
        },
    },
    ensure_ascii=False,
    indent=2,
)


class LLMIntentClassifier:
    """
    调用 LLM 做高层意图识别，输出结构化 JSON，辅助智能源路由。
    """

    SYSTEM_PROMPT = (
        "你是一名严谨的意图分类助手。"
        "请阅读用户查询，判断其属于哪个业务意图，并严格输出 JSON。"
        "不得输出除 JSON 以外的内容。"
    )

    def __init__(
        self,
        client: Optional[HKGAIClient] = None,
        allowed_intents: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: Optional[float] = None,
        max_retries: int = 2,
        retry_on_low_confidence: bool = True,
    ):
        self.client = client or HKGAIClient()
        self.allowed_intents = allowed_intents or INTENT_DEFINITIONS
        self.intent_names = {item["name"] for item in self.allowed_intents}

        env_threshold = os.getenv("LLM_INTENT_CONFIDENCE")
        if confidence_threshold is not None:
            threshold = confidence_threshold
        elif env_threshold:
            try:
                threshold = float(env_threshold)
            except ValueError:
                threshold = 0.55
        else:
            threshold = 0.55

        self.confidence_threshold = threshold
        self.max_retries = max(1, max_retries)
        self.retry_on_low_confidence = retry_on_low_confidence
        self.low_confidence_instruction = (
            "第一次判定置信度偏低，请重新审视用户意图。若无法确定，返回 rag。"
        )
        self.last_result: Optional[Dict[str, Any]] = None

    def _build_user_prompt(
        self,
        query: str,
        extra_instruction: Optional[str] = None,
        attempt: int = 0,
    ) -> str:
        intent_lines = []
        for item in self.allowed_intents:
            examples = "；".join(item.get("example_queries", []))
            intent_lines.append(
                f"- {item['name']}: {item['description']} 示例：{examples or '无'}"
            )

        instructions = "\n".join(intent_lines)
        prompt = (
            f"用户问题：{query}\n\n"
            "意图选项说明：\n"
            f"{instructions}\n\n"
            "请根据以下 JSON 模板返回：\n"
            f"{SCHEMA_DESCRIPTION}\n"
            "必须保证是有效 JSON，字段齐全。"
        )
        if extra_instruction:
            prompt += f"\n\n额外要求：{extra_instruction}"
        if attempt > 0:
            prompt += (
                f"\n\n注意：此前解析失败或置信度不足，这是第 {attempt + 1} 次尝试，"
                "请更明确地判断意图并返回有效 JSON。"
            )
        return prompt

    @staticmethod
    def _temperature_for_attempt(attempt: int) -> float:
        if attempt <= 0:
            return 0.0
        return min(0.1 * attempt + 0.1, 0.6)

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None

        candidates = [content]

        # 去掉围栏
        if "```" in content:
            fenced = "\n".join(
                line for line in content.splitlines() if not line.strip().startswith("```")
            )
            candidates.append(fenced)

        # 提取首个 JSON
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                return parsed
            except json.JSONDecodeError:
                continue

        logging.warning("LLMIntentClassifier: 无法解析 JSON 响应：%s", content)
        return None

    def classify(
        self,
        query: str,
        extra_instruction: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        parsed: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            user_prompt = self._build_user_prompt(
                query,
                extra_instruction=extra_instruction,
                attempt=attempt,
            )
            response = self.client.chat(
                self.SYSTEM_PROMPT,
                user_prompt,
                max_tokens=400,
                temperature=self._temperature_for_attempt(attempt),
            )

            if not response:
                last_error = "LLM 返回空响应"
                continue

            if "error" in response:
                last_error = response["error"]
                continue

            parsed = self._parse_response(response.get("content", ""))
            if parsed:
                break

        if not parsed and last_error:
            logging.warning("LLMIntentClassifier: LLM 请求失败：%s", last_error)
        elif not parsed:
            logging.warning("LLMIntentClassifier: 多轮尝试仍无法解析 JSON")

        self.last_result = parsed
        return parsed

    def select_intent(self, query: str) -> Optional[str]:
        def extract(result: Optional[Dict[str, Any]]) -> (Optional[str], float):
            if not result:
                return None, 0.0
            intent_value = result.get("intent")
            if intent_value not in self.intent_names:
                logging.info("LLMIntentClassifier: 未知 intent=%s，fallback", intent_value)
                return None, 0.0
            try:
                confidence_value = float(result.get("confidence", 0))
            except (TypeError, ValueError):
                confidence_value = 0.0
            return intent_value, confidence_value

        result = self.classify(query)
        intent, confidence = extract(result)

        if intent and confidence >= self.confidence_threshold:
            return intent

        if self.retry_on_low_confidence:
            logging.info(
                "LLMIntentClassifier: intent=%s 置信度 %.2f 低于阈值 %.2f，重新尝试",
                intent,
                confidence,
                self.confidence_threshold,
            )
            retry_result = self.classify(
                query,
                extra_instruction=self.low_confidence_instruction,
            )
            intent_retry, confidence_retry = extract(retry_result)
            if intent_retry and confidence_retry >= self.confidence_threshold:
                return intent_retry
            intent = intent_retry
            confidence = confidence_retry

        if intent:
            logging.info(
                "LLMIntentClassifier: intent=%s 置信度 %.2f 仍低于阈值 %.2f，fallback",
                intent,
                confidence,
                self.confidence_threshold,
            )
        return None

