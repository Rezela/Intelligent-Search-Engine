from api import HKGAIClient
from retrieve import chromadb_retrieve
from rerank import rerank
from typing import Tuple
import time
import logging
from logs import init_logger, new_query_id

# Prompt 构造
def make_prompt(query: str, context: str) -> Tuple[str, str]:
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



class FullRAG:
    def __init__(self, db):
        self.db = db
        self.llm_client = HKGAIClient()

    def query(self, user_query: str, language: str = "Chinese", top_k: int = 5):
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
        system_prompt, user_prompt = make_prompt(user_query, context)
        result = self.llm_client.chat(system_prompt, user_prompt, max_tokens=256, temperature=0.0)
        t3 = time.time()

        # Step 5: 日志
        retrieval_time = t1 - t0
        rerank_time = t2 - t1
        llm_time = t3 - t2
        total_time = t3 - t0

        logging.info(f"Query: {user_query}")
        logging.info(f"Retrieved: {retrieved_results}")
        logging.info(f"Reranked: {reranked_results}")
        logging.info(f"LLM Answer: {result['content']}")
        logging.info(f"Times [retrieval={retrieval_time:.3f}s, rerank={rerank_time:.3f}s, llm={llm_time:.3f}s, total={total_time:.3f}s]")

        return {
            "query": user_query,
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
    query = "哆啦A梦使用的3个秘密道具分别是什么？"
    result = rag.query(query, language="Chinese")


    print("=== Retrieved ===")
    for i, (chunk, score) in enumerate(result["retrieved"], 1):
        print(f"[{i}] (retrieval_score={score:.3f}) {chunk}")

    print("\n=== Reranked ===")
    for i, (chunk, score) in enumerate(result["reranked"], 1):
        print(f"[{i}] (rerank_score={score:.3f}) {chunk}")

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Timing ===")
    print(result["timing"])