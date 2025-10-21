from typing import List

# 定长分片
def split_into_chunks(docs: str, chunk_size: int = 1000) -> List[str]:
    chunks = []
    with open(docs, 'r', encoding = 'utf-8') as file:
        content = file.read()
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i+chunk_size])
    return chunks

def split_into_paragraphs(docs: str, max_chunk_size: int = 1000) -> List[str]:
    chunks = []
    with open(docs, 'r', encoding = 'utf-8') as file:
        content = file.read()
    paragraphs = content.split("\n\n")
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            # 如果段落长度超过最大分块大小，则进行分块
            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i+max_chunk_size])
        else:
            chunks.append(paragraph)
    return chunks


if __name__ == "__main__":
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    print("定长分块: \n")
    chunks1 = split_into_chunks(docs, chunk_size)
    for idx, chunk in enumerate(chunks1, 1):
        print(f"Chunk {idx}:\n{chunk}\n{'-'*40}")

    print("\n段落分块: \n")
    chunks2 = split_into_paragraphs(docs, max_chunk_size = 300)
    for idx ,chunk in enumerate(chunks2, 1):
        print(f"Paragraph Chunk {idx}:\n{chunk}\n{'-'*40}")