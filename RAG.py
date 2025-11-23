import json
import time
import logging
from typing import Tuple
import json
from api import HKGAIClient
from retrieve import chromadb_retrieve
from rerank import rerank
from logs import init_logger, new_query_id
from SourceRouter import SourceRouter
from SourceAPI import HANDLERS
from intent_classifier import LLMIntentClassifier
from query_optimze import DeepSeekRAGOptimizer, preprocess_query_with_deepseek
from typing import Tuple, Dict, Any, List
# RAG 的 Prompt 构造 （仅使用本地上下文，无信息则不输出）
def make_rag_prompt(query: str, context: str) -> Tuple[str, str]:
    system = (
        "You are a careful assistant. Use only the provided context to answer. "
        "If the answer is not in context, say you don't know."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context (retrieved):\n{context}\n\n"
        "Answer with sources if relevant."
    )
    return system, user

# 调用 API 的 prompt 构造
def _format_api_context(context):
    if isinstance(context, dict):
        lines = []
        summary = context.get("summary")
        if summary:
            lines.append(summary)

        if context.get("mode"):
            lines.append(f"出行方式: {context['mode']}")

        if "distance" in context and "duration" in context:
            lines.append(f"路程 {context['distance']} / 预计用时 {context['duration']}")

        location = context.get("location")
        if location:
            loc_desc = f"{location.get('name')} ({location.get('lat')}, {location.get('lon')})"
            lines.append(f"位置: {loc_desc}")

        current = context.get("current") or {}
        temp = current.get("temperature")
        humidity = current.get("humidity")
        if temp is not None or humidity is not None:
            lines.append(
                f"当前温度 {temp}°C, 湿度 {humidity}%" if humidity is not None else f"当前温度 {temp}°C"
            )

        hourly = context.get("forecast_hourly") or context.get("forecast") or []
        if hourly:
            first = hourly[0]
            lines.append(
                f"短期预报: {first.get('description')}，温度 {first.get('temperature')}°C，降水概率 {int((first.get('precip_probability') or 0)*100)}%"
            )

        daily = context.get("forecast_daily") or []
        if daily:
            day = daily[0]
            lines.append(
                f"明日 {day.get('date')}：{day.get('description')}，温度 {day.get('temp_min')}~{day.get('temp_max')}°C"
            )

        air_block = context.get("air_quality") or {}
        air_current = air_block.get("current") if isinstance(air_block, dict) else air_block
        if air_current and air_current.get("aqi"):
            lines.append(f"空气质量指数 AQI={air_current['aqi']}（{air_current.get('category')}）")
        air_forecast = air_block.get("forecast") if isinstance(air_block, dict) else None
        if air_forecast and air_forecast.get("aqi"):
            lines.append(f"未来空气质量：AQI={air_forecast['aqi']}（{air_forecast.get('category')}）")

        typhoon_signal = context.get("typhoon_signal")
        if isinstance(typhoon_signal, dict) and typhoon_signal.get("message"):
            lines.append(f"热带气旋信号：{typhoon_signal.get('message')}")

        warnings = context.get("warnings") or []
        if warnings:
            lines.append("天气警告：" + "；".join(str(w) for w in warnings[:3]))

        uv = context.get("uv_index")
        if isinstance(uv, dict) and uv.get("value") is not None:
            lines.append(f"紫外线指数：{uv.get('value')}（{uv.get('desc') or '—'}）")

        tc_track = context.get("tc_track")
        if isinstance(tc_track, dict):
            latest = tc_track.get("latest") or {}
            storm = tc_track.get("stormName") or tc_track.get("stormId")
            if latest and storm:
                lines.append(
                    f"{storm} 当前位置 {latest.get('lat')}, {latest.get('lon')} 时间 {latest.get('time')}，强度 {latest.get('intensity') or latest.get('category')}"
                )

        tips = context.get("special_weather_tips") or []
        if tips:
            first_tip = tips[0]
            if isinstance(first_tip, dict):
                content = first_tip.get("content") or first_tip.get("title")
            else:
                content = first_tip
            if content:
                lines.append(f"特别天气提示：{content}")

        advisories = context.get("advisories") or []
        if advisories:
            lines.append("提示：" + "；".join(advisories))

        steps = context.get("steps") or []
        if steps:
            preview = "; ".join(
                f"{step.get('instruction')} ({step.get('distance')})" for step in steps[:3]
            )
            lines.append(f"路线提示: {preview}")

        meta = context.get("meta") or {}
        if meta:
            strategy = meta.get("strategy")
            query_used = meta.get("query")
            if strategy:
                lines.append(f"搜索策略：{strategy}")
            if query_used:
                lines.append(f"使用查询：{query_used}")

        items = context.get("items") or []
        if items:
            previews = []
            for item in items[:3]:
                title = item.get("title") or "无标题"
                link = item.get("link") or ""
                snippet = (item.get("snippet") or "").strip()
                previews.append(f"{title} - {snippet} ({link})")
            lines.append("搜索结果：" + "； ".join(previews))

        return "\n".join(lines) or json.dumps(context, ensure_ascii=False)
    return str(context)


def _stringify_context(context):
    def convert(obj, depth=0):
        if isinstance(obj, list):
            sliced = [convert(item, depth + 1) for item in obj[:5]]
            if len(obj) > 5:
                sliced.append(f"...共省略 {len(obj) - 5} 条")
            return sliced
        if isinstance(obj, dict):
            trimmed = {}
            for k, v in obj.items():
                if k == "raw":
                    continue
                trimmed[str(k)] = convert(v, depth + 1)
            return trimmed
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
        return obj

    if isinstance(context, dict):
        safe_context = convert(context)
        return json.dumps(safe_context, ensure_ascii=False, indent=2)
    return str(context)


def make_api_prompt(query: str, context) -> Tuple[str, str]:
    formatted_context = _format_api_context(context)
    system = (
        "You are a helpful assistant. Use the provided API result as the authoritative source. "
        "Do not hallucinate data that is not included."
    )
    user = (
        f"User asked:\n{query}\n\n"
        f"API summary:\n{formatted_context}\n\n"
        "Answer clearly and concisely."
    )
    return system, user

class FullRAG:
    def __init__(self, db):
        self.db = db
        self.llm_client = HKGAIClient()
        self.intent_classifier = LLMIntentClassifier(client=self.llm_client)
        self.router = SourceRouter(intent_classifier=self.intent_classifier)

        try:
            self.optimizer = DeepSeekRAGOptimizer("sk-009b68ac7a984590bf76912a64d85990")
            print("✅ DeepSeek查询优化器已初始化")
        except Exception as e:
            print(f"❌ 优化器初始化失败: {e}")
            self.optimizer = None



    def _optimize_query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        if self.optimizer is None:
            return user_query, {
                "original_question": user_query,
                "optimized_question": user_query,
                "success": False,
                "optimization_notes": "优化器未启用"
            }
    
        try:
            optimization_result = self.optimizer.optimize_question(user_query)
            optimized_query = optimization_result["optimized_question"]
        
        # 记录优化信息
            logging.info("==== Query Optimization ====")
            logging.info(f"Original: {user_query}")
            logging.info(f"Optimized: {optimized_query}")
            logging.info(f"Key Entities: {optimization_result.get('key_entities', [])}")
            logging.info(f"Search Keywords: {optimization_result.get('search_keywords', [])}")
            logging.info(f"RAG Suitability Score: {optimization_result.get('rag_suitability_score', 'N/A')}")
            logging.info(f"Optimization Notes: {optimization_result.get('optimization_notes', 'N/A')}")
        
            return optimized_query, optimization_result
        
        except Exception as e:
            logging.warning(f"查询优化失败: {e}")
            return user_query, {
                "original_question": user_query,
                "optimized_question": user_query,
                "success": False,
                "error": str(e)
            }
    def query(self, user_query: str, language: str = "Chinese", top_k: int = 5):
        # Step 0: 智能源选择
        t0 = time.time()
        optimized_query, optimization_result = self._optimize_query(user_query)
        t_optimize = time.time() - t0
        source = self.router.route(user_query)
        intent_snapshot = self.intent_classifier.last_result
        if intent_snapshot:
            try:
                logging.info(
                    "Intent Result: %s",
                    json.dumps(intent_snapshot, ensure_ascii=False),
                )
            except TypeError:
                logging.info("Intent Result (non-serializable): %s", intent_snapshot)

        if source in HANDLERS:
            context = HANDLERS[source].handle(user_query, metadata=intent_snapshot)
            system_prompt, user_prompt = make_api_prompt(user_query, context)
            result = self.llm_client.chat(system_prompt, user_prompt)
            t1 = time.time()
            api_source_time = t1 - t0

            logging.info("==== New Query ====")
            logging.info(f"Query: {user_query}")
            logging.info(f"Source: {source}")
            logging.info(f"API Context: {_stringify_context(context)}")
            logging.info(f"LLM Answer: {result['content']}")
            logging.info(f"Times [api={api_source_time:.3f}s]")

            return {
                "query": user_query,
                "source": source,
                "context": context,
                "answer": result["content"],
                "timing": {
                    "api": api_source_time
                }
            }

        # 默认走 RAG
        return self._rag_pipeline(user_query, language, top_k)


    def _rag_pipeline(self, user_query: str, language: str = "Chinese", top_k: int = 5):
        # Step 1: 检索
        t0 = time.time()
        retrieved_results = chromadb_retrieve(self.db, user_query, language, top_k=top_k)
        t1 = time.time()

        # Step 2: rerank
        reranked_results = rerank(user_query, [doc for doc, _ in retrieved_results], top_k=3)  # [(doc, score)]
        t2 = time.time()

        # Step 3: 构造上下文
        context = "\n\n".join([chunk for chunk, score in reranked_results])

        # Step 4: 调用 LLM
        system_prompt, user_prompt = make_rag_prompt(user_query, context)
        result = self.llm_client.chat(system_prompt, user_prompt, max_tokens=256, temperature=0.0)
        t3 = time.time()

        # Step 5: 日志
        retrieval_time = t1 - t0
        rerank_time = t2 - t1
        llm_time = t3 - t2
        total_time = t3 - t0

        logging.info("==== New Query ====")  # 分隔符
        logging.info(f"Query: {user_query}")
        logging.info(f"Retrieved: {retrieved_results}")
        logging.info(f"Reranked: {reranked_results}")
        logging.info(f"LLM Answer: {result['content']}")
        logging.info(f"Times [retrieval={retrieval_time:.3f}s, rerank={rerank_time:.3f}s, llm={llm_time:.3f}s, total={total_time:.3f}s]")

        return {
            "query": user_query,
            "source": "rag",
            "retrieved": retrieved_results,  # [(doc, score)]
            "reranked": reranked_results,
            "answer": result["content"],
            "timing": {
                "retrieval": retrieval_time,
                "rerank": rerank_time,
                "llm": llm_time,
                "total": total_time
            }
        }

if __name__ == "__main__":
    from DB import get_db

    # 初始化日志系统
    log_filename = init_logger()
    print(f"Log saved to: {log_filename}")

    # 生成唯一查询 ID
    query_id = new_query_id()
    print(f"Query ID: {query_id}")

    # 使用持久化数据库
    db = get_db(persistent=True, path="./chroma_db", name="default")

    
    # 初始化RAG系统（传入API密钥）
    rag = FullRAG(db)

    # 测试不同 query
    queries = [
        "今天关于OpenAI的最新新闻",
    ]

    for q in queries:
        print("\n\n=== Query ===")
        print(q)
        result = rag.query(q, language="Chinese")

        # 显示优化信息（新增）
        if "optimization_info" in result:
            print("\n=== Optimization Info ===")
            opt_info = result["optimization_info"]
            print(f"Original: {opt_info.get('original_question')}")
            print(f"Optimized: {opt_info.get('optimized_question')}")
            print(f"Success: {opt_info.get('success')}")
            print(f"Notes: {opt_info.get('optimization_notes')}")

        # if "retrieved" in result and "reranked" in result:
        #     print("\n=== Retrieved ===")
        #     for i, (chunk, score) in enumerate(result["retrieved"], 1):
        #         print(f"[{i}] (retrieval_score={score:.3f}) {chunk}")

        #     print("\n=== Reranked ===")
        #     for i, (chunk, score) in enumerate(result["reranked"], 1):
        #         print(f"[{i}] (rerank_score={score:.3f}) {chunk}")

        #     print("\n=== Answer ===")
        #     print(result["answer"])
        # else:
        #     print("\n=== API Result ===")
        #     print(_stringify_context(result.get("context", "")))
        #     print("\n=== Answer ===")
        #     print(result.get("answer", ""))

        # print("\n=== Timing ===")
        # for k, v in result["timing"].items():
        #     print(f"{k}: {v:.3f}s")