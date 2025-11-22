import argparse

from DB import get_db
from RAG import FullRAG
from logs import init_logger, new_query_id


def run_queries(language: str):
    log_filename = init_logger()
    print(f"Log saved to: {log_filename}")

    query_id = new_query_id()
    print(f"Query ID: {query_id}")

    db = get_db(persistent=True, path="./chroma_db", name="default")
    rag = FullRAG(db)

    queries = [
        "哆啦A梦使用的3个秘密道具分别是什么？",  # RAG
        "北京今天的天气情况",  # Weather API
        "科大到中环要多久",  # Traffic API
        "中国石化今天的收盘价是多少",  # Finance API
        "今天关于OpenAI的最新新闻",  # Google Search API
    ]

    for q in queries:
        print("\n\n=== Query ===")
        print(q)
        result = rag.query(q, language=language)

        if "retrieved" in result and "reranked" in result:
            print("\n=== Retrieved ===")
            for i, (chunk, score) in enumerate(result["retrieved"], 1):
                print(f"[{i}] (retrieval_score={score:.3f}) {chunk}")

            print("\n=== Reranked ===")
            for i, (chunk, score) in enumerate(result["reranked"], 1):
                print(f"[{i}] (rerank_score={score:.3f}) {chunk}")

            print("\n=== Answer ===")
            print(result["answer"])
        else:
            print("\n=== API Result ===")
            print(result.get("context", ""))
            print("\n=== Answer ===")
            print(result.get("answer", ""))

        print("\n=== Timing ===")
        for k, v in result["timing"].items():
            print(f"{k}: {v:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FullRAG queries.")
    parser.add_argument("--language", default="Chinese", help="Query language.")
    args = parser.parse_args()
    run_queries(args.language)

