from typing import Iterable, List, Tuple
from embedding import embed_chunk

# retrieve from chromadb
def chromadb_retrieve(collections, query: str, langauge: str = 'Chinese', top_k: int = 3) -> List[Tuple[str, float]]:
    """
    在一个或多个 ChromaDB 集合中检索与 query 最相关的 top_k 文档。
    collections 可以是单个 collection 或者 collection 列表。
    """
    if not isinstance(collections, (list, tuple)):
        collections = [collections]

    query_embedding = embed_chunk(query, langauge)
    combined = []
    for collection in collections:
        if collection is None:
            continue
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        combined.extend(zip(documents, distances))

    combined.sort(key=lambda item: item[1])
    return combined[:top_k]

if __name__ == '__main__':

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

    user_query = "哆啦A梦使用的3个秘密道具分别是什么？"
    retrieved_chunks = chromadb_retrieve(db, user_query, 'Chinese', top_k=5)
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")