from typing import List
from embedding import embed_chunk

# retrieve from chromadb
def chromadb_retrieve(db, query: str, langauge: str = 'Chinese', top_k: int = 3) -> List[str]:
    """
    在指定的 ChromaDB 集合中检索与 query 最相关的 top_k 文档
    :param db: chromadb.Collection 对象
    :param query: 用户查询字符串
    :param top_k: 返回的文档数量
    :return: 文本列表
    """
    query_embedding = embed_chunk(query, langauge)
    # 使用chromadb_collection集合来检索query
    results = db.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]  # [0]: 取第一个查询的结果（即唯一的query）

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