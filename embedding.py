from chunk import split_into_sentences
from sentence_transformers import SentenceTransformer
from typing import List

english_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chinese_embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

def embed_chunk(chunk: str, language : str= 'English') -> List[float]:
    if language == 'English':
        embed = english_embedding_model.encode(chunk, normalize_embedding=True)
    elif language == 'Chinese':
        embed = chinese_embedding_model.encode(chunk, normalize_embedding=True)
    else:
        # 处理不支持的语言，使用默认的英文模型
        embed = english_embedding_model.encode(chunk, normalize_embedding=True)
    return embed.tolist()