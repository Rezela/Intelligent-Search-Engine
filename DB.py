import chromadb
from typing import List

# 使用chromadb 保存chunks和embeddings向量 到数据库集合
def save_embeddings(db, chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]  # id: 索引
        )

if __name__ == '__main__':

    from chunk import split_into_sentences
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    chunks = split_into_sentences(docs, max_chunk_size=chunk_size, language='Chinese')

    from embedding import embed_chunk
    embeddings = []
    for idx, chunk in enumerate(chunks, 1):
        embeddings.append(embed_chunk(chunk, 'Chinese'))

    chromadb_client = chromadb.EphemeralClient()  # 创建一个临时的ChromaDB客户端,临时 ChromaDB 客户端，数据只存在于内存中，程序结束后数据会消失。
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")  # 在数据库里获取或新建一个名为 "default" 的集合

    # 删除名为"default"的集合
    # chromadb_client.delete_collection(name="default")

    db = chromadb_collection
    save_embeddings(db, chunks, embeddings)

    print("database count: ", db.count())
    results = db.get()
    print(results.keys())  # 查看包含哪些字段
    print(results["ids"])  # 打印所有 ID
    print(results["documents"][:3])  # 打印前 3 个文档

    result = db.get(ids=["2"])
    print(result["documents"])
