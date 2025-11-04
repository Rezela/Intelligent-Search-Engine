import re
from typing import List
import spacy
# 需要先运行: pip install spacy
# 或: conda install -c conda-forge spacy

# 定长分片
def split_into_chunks(docs: str, chunk_size: int = 1000) -> List[str]:
    chunks = []
    with open(docs, 'r', encoding='utf-8') as file:
        content = file.read()
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i + chunk_size])
    return chunks


# 段落分片
def split_into_paragraphs(docs: str, max_chunk_size: int = 1000) -> List[str]:
    chunks = []
    with open(docs, 'r', encoding='utf-8') as file:
        content = file.read()
    paragraphs = content.split("\n\n")  # 双换行：自然段落
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            # 如果段落长度超过最大分块大小，则进行分块
            for i in range(0, len(paragraph), max_chunk_size):
                chunks.append(paragraph[i:i + max_chunk_size])
        else:
            chunks.append(paragraph)
    return chunks


# 句子分片
# nlp_english = spacy.load("en_core_web_sm")  # 加载英文模型
# nlp_chinese = spacy.load("zh_core_web_sm")  # 加载中文模型
# ！！！需要先运行: pip install spacy
# 或: conda install -c conda-forge spacy
# 随后安装python -m spacy download en_core_web_sm
# python -m spacy download zh_core_web_sm


def split_into_sentences(docs: str, max_chunk_size: int = 100, language : str= 'English') -> List[str]:
    """
    按句子分片，优先保持语义完整。
    如果句子过长，再进行二次切割。
    """
    chunks = []
    with open(docs, 'r', encoding='utf-8') as file:
        content = file.read()


    if language == 'English':
        nlp_english = spacy.load("en_core_web_sm")  # 加载英文模型
        doc = nlp_english(content)  # 使用英文模型
    elif language == 'Chinese':
        nlp_chinese = spacy.load("zh_core_web_sm")  # 加载中文模型
        doc = nlp_chinese(content)  # 使用中文模型

    current_chunk = ""
    for sent in doc.sents:  # 遍历句子
        sent_text = sent.text.strip()  # strip()去除句子前后的空格
        if len(current_chunk) + len(sent_text) <= max_chunk_size:
            current_chunk += " " + sent_text
            # 如果未超限，将句子用空格连接到current_chunk，尽量让每个块包含尽可能多的完整句子，保证语义完整性
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())  # 已经累积好的current_chunk作为一个完整块存入chunks
            if len(sent_text) > max_chunk_size:  # 当前sent_text自身就过长，在句子内细切
                sub_sents = re.split(r'. |，|。|！|？', sent_text)
                current = ""
                for i in range(len(sub_sents)):
                    if len(current) + len(sub_sents) <= max_chunk_size:
                        current += " " + sub_sents[i]
                    else:
                        chunks.append(current.strip())
                        current = sub_sents[i]
                if current:
                    chunks.append(current.strip())
                '''
                for i in range(0, len(sent_text), max_chunk_size):
                    chunks.append(sent_text[i:i+max_chunk_size])
                current_chunk = ""  # 重置current_chunk
                '''
            else:
                current_chunk = sent_text  # 当前sent_text足够短，直接作为新块起点
    if current_chunk:
        chunks.append(current_chunk.strip())  # 提交尾部块
    return chunks

if __name__ == "__main__":
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    print("定长分块: \n")
    chunks1 = split_into_chunks(docs, chunk_size)
    for idx, chunk in enumerate(chunks1, 1):
        print(f"Chunk {idx}:\n{chunk}\n{'-' * 40}")

    print("\n段落分块: \n")
    chunks2 = split_into_paragraphs(docs, max_chunk_size=300)
    for idx, chunk in enumerate(chunks2, 1):
        print(f"Paragraph Chunk {idx}:\n{chunk}\n{'-' * 40}")

    print("\n句子分块： \n")
    chunks3 = split_into_sentences(docs, max_chunk_size=100, language='Chinese')
    for idx, chunk in enumerate(chunks3, 1):
        print(f"Sentence Chunk {idx}:\n{chunk}\n{'-' * 40}")