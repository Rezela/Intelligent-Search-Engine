import json
import time
import logging
from typing import Tuple
from api import HKGAIClient
from retrieve import chromadb_retrieve
from rerank import rerank
from logs import init_logger, new_query_id
from SourceRouter import SourceRouter
from SourceAPI import HANDLERS
from intent_classifier import LLMIntentClassifier

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
def make_api_prompt(query: str, context: str) -> Tuple[str, str]:
    system = "You are a helpful assistant." \
             "Use the provided API result as the authoritative source to answer the question."
    user = f"User asked:\n{query}\n\nAPI result:\n{context}\n\nAnswer clearly and concisely."
    return system, user

class FullRAG:
    def __init__(self, db):
        self.db = db
        self.llm_client = HKGAIClient()
        self.intent_classifier = LLMIntentClassifier(client=self.llm_client)
        self.router = SourceRouter(intent_classifier=self.intent_classifier)

    def query(self, user_query: str, language: str = "Chinese", top_k: int = 5):
        # Step 0: 智能源选择
        t0 = time.time()
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
            context = HANDLERS[source].handle(user_query)
            system_prompt, user_prompt = make_api_prompt(user_query, context)
            result = self.llm_client.chat(system_prompt, user_prompt)
            t1 = time.time()
            api_source_time = t1 - t0

            logging.info("==== New Query ====")
            logging.info(f"Query: {user_query}")
            logging.info(f"Source: {source}")
            logging.info(f"API Context: {context}")
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

    # 假设已经 save_embeddings(db, chunks, embeddings)
    rag = FullRAG(db)

    # 测试不同 query
    queries = [
        "哆啦A梦使用的3个秘密道具分别是什么？",   # RAG
        "北京今天的天气情况",                   # Weather API
        "科大到中环要多久",                      # Traffic API
        "中国石化今天的收盘价是多少",             # Finance API
        

    ]

    for q in queries:
        print("\n\n=== Query ===")
        print(q)
        result = rag.query(q, language="Chinese")

        if result["source"] in ["weather_api", "traffic_api", "finance_api", "public_service_api"]:
            print("\n=== API Result ===")
            print(result["context"])
            print("\n=== Answer ===")
            print(result["answer"])
        else:
            print("\n=== Retrieved ===")
            for i, (chunk, score) in enumerate(result["retrieved"], 1):
                print(f"[{i}] (retrieval_score={score:.3f}) {chunk}")

            print("\n=== Reranked ===")
            for i, (chunk, score) in enumerate(result["reranked"], 1):
                print(f"[{i}] (rerank_score={score:.3f}) {chunk}")

            print("\n=== Answer ===")
            print(result["answer"])

        print("\n=== Timing ===")
        for k, v in result["timing"].items():
            print(f"{k}: {v:.3f}s")