import json
import time
import logging
import re
from typing import Any, Dict, Optional

from SourceAPI import HANDLERS


class DeepSearchManager:
    """
    管理多轮搜索/回退策略：
    1. 首轮使用现有 RAG/API;
    2. 若自评失败则尝试 Google 搜索补充;
    3. 最多尝试 max_attempts 轮，超出则礼貌失败。
    """

    def __init__(self, rag_engine, max_attempts: int = 3):
        self.rag = rag_engine
        self.max_attempts = max(1, max_attempts)
        self.llm_client = rag_engine.llm_client
        self.google_handler = HANDLERS.get("google_search_api")
        self.logger = logging.getLogger(__name__)

    def _extract_json_from_markdown(self, content: str) -> str:
        """
        从 markdown 代码块中提取 JSON 内容
        支持 ```json 和 ``` 格式
        """
        # 匹配 ```json 或 ``` 包围的 JSON 内容
        json_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 如果没有代码块标记，尝试直接返回内容
        return content.strip()

    def run(self, query: str, language: str = "Chinese") -> Dict[str, Any]:
        attempt_history = []
        strategy = "primary"
        metadata = None

        for attempt in range(1, self.max_attempts + 1):
            if strategy == "google":
                result = self._answer_with_google(query, metadata)
            else:
                result = self.rag.query(query, language=language)

            metadata = getattr(self.rag.intent_classifier, "last_result", None)
            attempt_history.append(
                {
                    "attempt": attempt,
                    "strategy": strategy,
                    "source": result.get("source"),
                    "answer_preview": (result.get("answer") or "")[:120],
                }
            )

            evaluation = self._self_evaluate(query, result)
            result["self_evaluation"] = evaluation
            result["attempts_used"] = attempt

            if evaluation.get("satisfied") or attempt == self.max_attempts:
                result["attempt_history"] = attempt_history
                return result

            strategy = self._decide_next_step(evaluation, result, attempt)
            if strategy == "stop":
                result["attempt_history"] = attempt_history
                return result

        result["attempt_history"] = attempt_history
        return result

    # ---------------- Internal helpers ---------------- #

    def _self_evaluate(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        answer = result.get("answer") or ""
        source = result.get("source") or "unknown"
        context = {
            "question": query,
            "answer": answer,
            "source": source,
        }
        system = (
            "你是回答质量检测器。请严格判断模型回答是否满足提问，不要编造。"
            "如果答案缺乏关键信息或只有“我不知道”，就标记为不满足。"
        )
        user = (
            "请阅读以下问答，并输出 JSON：\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "输出格式：{\n"
            '  "satisfied": true/false,\n'
            '  "reason": "简要原因",\n'
            '  "confidence": 0-1,\n'
            '  "next_step": "google" | "rag" | "stop"\n'
            "}\n"
            "如果无法解析，则根据经验给出 best-effort 判断。"
        )
        resp = self.llm_client.chat(system, user, max_tokens=200, temperature=0.0)
        content = resp.get("content") or ""
        try:
            # 清理 markdown 代码块标记
            json_content = self._extract_json_from_markdown(content)
            data = json.loads(json_content)
            data["satisfied"] = bool(data.get("satisfied"))
            data["confidence"] = float(data.get("confidence", 0))
            return data
        except Exception:
            self.logger.warning("Self-eval JSON 解析失败，fallback 原文: %s", content)
            return {
                "satisfied": False,
                "reason": "解析失败，保守认为答案不充分。",
                "confidence": 0.0,
                "next_step": "google",
            }

    def _decide_next_step(
        self, evaluation: Dict[str, Any], result: Dict[str, Any], attempt: int
    ) -> str:
        hint = (evaluation.get("next_step") or "").lower()
        if hint in {"stop", "fail", "exit"}:
            return "stop"

        if attempt >= self.max_attempts:
            return "stop"

        if hint in {"google", "search"} and self.google_handler:
            return "google"

        # 默认策略：首轮失败 → Google，第二轮失败 → 回 RAG（若可）
        if attempt == 1 and self.google_handler:
            return "google"

        if result.get("source") == "google_search_api":
            return "primary"

        return "google" if self.google_handler else "primary"

    def _answer_with_google(
        self, query: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.google_handler:
            return {
                "query": query,
                "source": "google_search_api",
                "answer": "无法使用 Google Search（未配置 API Key）。",
                "context": {"summary": None},
                "timing": {},
            }

        t0 = time.time()
        summary = self.google_handler.handle(query, metadata=metadata)
        t1 = time.time()
        summary_text = summary if isinstance(summary, str) else json.dumps(summary, ensure_ascii=False)

        system = (
            "You are a meticulous assistant. Use ONLY the provided Google search snippets "
            "to answer the user. Include key facts and keep the response concise."
        )
        user = (
            f"User question:\n{query}\n\n"
            f"Google search snippets:\n{summary_text}\n\n"
            "If the snippets do not contain the answer, say you cannot find it."
        )
        answer = self.llm_client.chat(system, user, max_tokens=256, temperature=0.2)

        return {
            "query": query,
            "source": "google_search_api",
            "context": {"summary": summary_text},
            "answer": answer.get("content", ""),
            "timing": {"google": t1 - t0},
        }

