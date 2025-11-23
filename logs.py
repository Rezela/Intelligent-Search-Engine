import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional


_LOGGER_NAME = "rag_app"


def _ensure_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def init_logger(log_name: str = "rag_engine") -> str:
    """初始化专用日志文件，仅记录重要事件。"""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir,
        f"{log_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = _ensure_logger()
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("Logs saved to %s", log_filename)
    return log_filename


def new_query_id() -> str:
    query_id = uuid.uuid4()
    logger = _ensure_logger()
    if logger.handlers:
        logger.info("Query ID: %s", query_id)
    return str(query_id)


def log_user_query(
    source: str,
    query: str,
    session_id: Optional[str] = None,
    has_image: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """记录来自 Web 或 CLI 的用户查询。"""
    logger = _ensure_logger()
    if not logger.handlers:
        return

    payload = {
        "source": source,
        "session": session_id,
        "has_image": has_image,
        "query": (query or "").strip(),
    }
    if extra:
        payload.update(extra)

    logger.info("USER_QUERY %s", json.dumps(payload, ensure_ascii=False))


def log_assistant_answer(
    source: str,
    answer: str,
    session_id: Optional[str] = None,
    timing: Optional[Dict[str, Any]] = None,
    context_preview: Optional[str] = None,
    attempt_history: Optional[Any] = None,
) -> None:
    """记录 LLM 最终回答及其耗时信息。"""
    logger = _ensure_logger()
    if not logger.handlers:
        return

    payload = {
        "source": source,
        "session": session_id,
        "answer_preview": (answer or "").strip()[:400],
    }
    if timing:
        payload["timing"] = timing
    if context_preview:
        payload["context"] = context_preview[:400]
    if attempt_history:
        payload["attempts"] = attempt_history

    logger.info("ASSISTANT_ANSWER %s", json.dumps(payload, ensure_ascii=False))