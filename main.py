import argparse
import base64
import json
from pathlib import Path

from DB import get_db
from RAG import FullRAG
from DeepSearch import DeepSearchManager
from logs import init_logger, new_query_id


def load_image_as_data_url(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"圖像文件不存在: {path}")

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    mime_type = mime_map.get(file_path.suffix.lower(), "image/png")

    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def run_queries(language: str):
    log_filename = init_logger()
    print(f"Log saved to: {log_filename}")

    query_id = new_query_id()
    print(f"Query ID: {query_id}")

    db = get_db(persistent=True, path="./chroma_db", name="default")
    rag = FullRAG(db)
    deep_search = DeepSearchManager(rag)

    queries = [
        {
            "text": "由堅尼地城前往香港國際機場的路線是什麼？",
        },
        {
            "text": "識別這座雕塑，解釋它的象徵意義，並告訴我它具體位於校園的哪個位置。",
            "image_path": "test/hkust.png",
        },
    ]

    for item in queries:
        if isinstance(item, dict):
            q = item["text"]
            image_path = item.get("image_path")
        else:
            q = item
            image_path = None

        image_data = None
        if image_path:
            try:
                image_data = load_image_as_data_url(image_path)
                print(f"[INFO] 已載入測試圖片: {image_path}")
            except Exception as exc:
                print(f"[WARN] 無法載入圖片 {image_path}: {exc}")

        print("\n\n=== Query ===")
        print(q)
        result = deep_search.run(q, language=language, image_data=image_data)

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

