# rag_minimal.py
# install FAISS by using the following command:
# pip install faiss-cpu sentence-transformers numpy
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from api import HKGAIClient
import json
import logging
from logging.handlers import RotatingFileHandler
import time

# 日志配置：最大1MB一个文件，最多保留5个
log_handler = RotatingFileHandler(
    "rag_engine.log",  # 日志文件名
    maxBytes=1*1024*1024,  # 单个日志文件最大1MB
    backupCount=5,  # 最多保留5个旧日志文件
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log_handler, logging.StreamHandler()]  # 同时输出到文件和控制台
)


# Minimal RAG pipeline
class MinimalRAG:
    def __init__(self, docs: List[str], embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # 1) Load embedder
        self.embedder = SentenceTransformer(embed_model_name)
        # 2) Build index
        self.docs = docs
        self.doc_embeddings = self._embed_texts(docs)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors + inner product
        # Normalize embeddings for cosine
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return vectors.astype("float32")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        q_vec = self._embed_texts([query])
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, top_k)
        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i != -1:
                results.append((int(i), float(s)))
        return results

    def build_context(self, query: str, top_k: int = 3) -> str:
        hits = self.search(query, top_k=top_k)
        parts = []
        for doc_idx, score in hits:
            parts.append(f"[Doc {doc_idx} | score={score:.3f}]\n{self.docs[doc_idx]}")
        return "\n\n".join(parts)


def make_prompt(query: str, context: str) -> Tuple[str, str]:
    system = (
        # Test the RAG only with the retrieved context
        "You are a careful assistant. Use only the provided context to answer. "
        "If the answer is not in context, say you don't know."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context (retrieved):\n{context}\n\n"
        "Answer with sources by citing Doc indices if relevant."
    )
    return system, user


def main():
    # Example documents (corpus)
    docs = [
        "Beijing is the capital of China. It is known for the Forbidden City and Tiananmen Square.",
        "Shanghai is a major financial hub in China, located on the eastern coast.",
        "Shenzhen is a technology and manufacturing center bordering Hong Kong.",
        "Hong Kong is a Special Administrative Region with a major port and financial industry.",
        "The Great Wall is a historic fortification stretching across northern China."
    ]

    rag = MinimalRAG(docs)
    client = HKGAIClient()
    query = "What is the capital of China?"
    logging.info(f"User query: {query}")  # 记录用户查询

    # 计时：检索
    t0 = time.time()
    context = rag.build_context(query, top_k=3)
    t1 = time.time()
    retrieval_time = t1 - t0
    logging.info(f"Retrieved context: {context}")  # 记录检索到的上下文
    logging.info(f"Retrieval time: {retrieval_time:.3f}s")

    # 计时：LLM
    system_prompt, user_prompt = make_prompt(query, context)
    t2 = time.time()
    result = client.chat(system_prompt, user_prompt, max_tokens=256, temperature=0.0)
    t3 = time.time()
    llm_time = t3 - t2
    total_time = t3 - t0

    logging.info(f"LLM raw response: {result['raw']}")
    logging.info(f"LLM final result: {result['content']}")  # 记录LLM的答案
    logging.info(f"LLM call time: {llm_time:.3f}s")
    logging.info(f"Total pipeline time: {total_time:.3f}s")

    print("=== Retrieved Context ===")
    print(context)
    print("\n=== LLM Answer ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[⏱️ Retrieval: {retrieval_time:.3f}s | LLM: {llm_time:.3f}s | Total: {total_time:.3f}s]")


if __name__ == "__main__":
    main()
