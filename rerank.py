from sentence_transformers import CrossEncoder
from typing import List

# 全局缓存cross_encoder模型
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        '''
        CrossEncoder:
            把 query 和 chunk 拼接后，一起输入同一个 Transformer。
            模型在编码时直接建模两者的交互关系，输出一个相关性分数。
            无需对两者embedding。
        '''
    return _cross_encoder

# rerank
def rerank(query: str, retrieved_chunks: List[str], top_k: int=3):
    cross_encoder = get_cross_encoder()
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
    scores = [score for _, score in scored_chunks[:top_k]]
    return chunks, scores

if __name__ =='__main__':

    from chunk import split_into_sentences
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    chunks = split_into_sentences(docs, max_chunk_size=chunk_size, language='Chinese')

    from embedding import embed_chunk
    embeddings = []
    for idx, chunk in enumerate(chunks, 1):
        embeddings.append(embed_chunk(chunk, 'Chinese'))

    import chromadb
    from DB import save_embeddings
    chromadb_client = chromadb.EphemeralClient()  # 创建一个临时的ChromaDB客户端,临时 ChromaDB 客户端，数据只存在于内存中，程序结束后数据会消失。
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")  # 在数据库里获取或新建一个名为 "default" 的集合
    # 删除名为"default"的集合
    # chromadb_client.delete_collection(name="default")
    db = chromadb_collection
    save_embeddings(db, chunks, embeddings)

    from retrieve import chromadb_retrieve
    user_query = "哆啦A梦使用的3个秘密道具分别是什么？"
    retrieved_chunks = chromadb_retrieve(db, user_query, 'Chinese', top_k=5)
    print("retrieved_chunks: \n")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")

    reranked_chunks = rerank(user_query, retrieved_chunks, top_k=3)
    print("reranked_chunks: \n")
    for i, chunk in enumerate(reranked_chunks):
        print(f"[{i}]: {chunk}\n")
