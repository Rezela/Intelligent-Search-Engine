import argparse
import json

from DB import get_db
from RAG import FullRAG
from DeepSearch import DeepSearchManager
from logs import init_logger, new_query_id
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def run_queries(language: str):
    log_filename = init_logger()
    print(f"Log saved to: {log_filename}")

    query_id = new_query_id()
    print(f"Query ID: {query_id}")

    db = get_db(persistent=True, path="./chroma_db", name="default")
    rag = FullRAG(db)
    deep_search = DeepSearchManager(rag)

    queries = [
        # "哆啦A梦使用的3个秘密道具分别是什么？",  # RAG
        
        # "北京今天的天气情况",  # Weather API
        # "What's the weather forecast for this afternoon in Hong Kong?",
        # "Will it rain in Shenzhen tomorrow?",
        # "What is the temperature in Beijing right now?",
        # "What time is sunset in Hong Kong today?",
        # "What is the wind speed in Shanghai?",
        # "明天广州的空气质量指数是多少？",
        # "澳門現在的濕度是多少？",
        # "今天香港的日出時間是幾點？",
        # "Is an evening run in Mong Kok today advisable?",

        # "香港天文臺現在懸掛的是什麼熱帶氣旋警告信號？",
        # "香港的公共圖書館在哪個熱帶氣旋警告信號下會關閉？",
        # "Latest HKO forecast track for nearest tropical cyclone",
        # "珠海今天的紫外線強度如何？",
        # "Assess the chance of Typhoon Signal No.8 tonight?",
        # "Will heavy rain affect Shenzhen Bay Port opening hours?",
        
        # "科大到中环要多久",  # Traffic API
        # "Provide the route from Kennedy Town to Hong Kong International Airport.",
        "由堅尼地城前往香港國際機場的路線是什麼？",
        # "What are the departure times for the Bus 91M from Diamond Hill station?",
        
        # "中国石化今天的收盘价是多少",  # Finance API
        # "Provide today’s Hang Seng Index percentage change at close.",
        # "現時金價是多少？",
        # "Compare the stock performance of NVIDIA (NVDA) and AMD over the last 5 days and summarize the top 3 reasons…",
        # "What is the current exchange rate between HKD and JPY, and how much is 50,000 Yen in HKD right now?",

        # "今天关于OpenAI的最新新闻",  # Google Search API
    ]

    for q in queries:
        print("\n\n=== Query ===")
        print(q)
        result = deep_search.run(q, language=language)
        if "optimization_info" in result:
            opt_info = result["optimization_info"]
            print("\n=== 查询优化信息 ===")
            print(f"原始查询: {opt_info.get('original_question', q)}")
            print(f"优化后查询: {opt_info.get('optimized_question', q)}")
            print(f"优化状态: {'✅ 成功' if opt_info.get('success') else '❌ 失败'}")
            if opt_info.get('optimization_notes'):
                print(f"优化说明: {opt_info.get('optimization_notes')}")
            if opt_info.get('key_entities'):
                print(f"关键实体: {opt_info.get('key_entities')}")
            if opt_info.get('search_keywords'):
                print(f"搜索关键词: {opt_info.get('search_keywords')}")
            if opt_info.get('rag_suitability_score'):
                print(f"RAG适用性评分: {opt_info.get('rag_suitability_score')}/10")
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
            context = result.get("context")
            if isinstance(context, dict):
                preview = context.get("summary")
                if not preview:
                    slim = {k: v for k, v in context.items() if k != "raw"}
                    preview = json.dumps(slim, ensure_ascii=False)  # need json module? not imported yet
                print(preview)
            else:
                print(context)
            print("\n=== Answer ===")
            print(result.get("answer", ""))

        timings = result.get("timing") or {}
        if timings:
            print("\n=== Timing ===")
            for k, v in timings.items():
                try:
                    print(f"{k}: {float(v):.3f}s")
                except (TypeError, ValueError):
                    print(f"{k}: {v}")

        if result.get("attempt_history"):
            print("\n=== Attempts ===")
            for info in result["attempt_history"]:
                print(
                    f"#{info['attempt']} strategy={info['strategy']} source={info['source']} preview={info['answer_preview']}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FullRAG queries.")
    parser.add_argument("--language", default="Chinese", help="Query language.")
    args = parser.parse_args()
    run_queries(args.language)

