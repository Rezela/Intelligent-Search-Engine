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