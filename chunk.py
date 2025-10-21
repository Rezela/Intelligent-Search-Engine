from typing import List

def split_into_chunks(docs: str, chunk_size: int = 1000) -> List[str]:
    chunks = []
    with open(docs, 'r', encoding = 'utf-8') as file:
        content = file.read()
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i+chunk_size])
    return chunks

if __name__ == "__main__":
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    chunks = split_into_chunks(docs, chunk_size)
    for idx, chunk in enumerate(chunks, 1):
        print(f"Chunk {idx}:\n{chunk}\n{'-'*40}")