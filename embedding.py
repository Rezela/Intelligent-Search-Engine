from sentence_transformers import SentenceTransformer
from typing import List

# english_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# chinese_embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')


# embedding模型全局缓存
embedding_cache = {}
embedding_models = {
    'English': "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensional
    'Chinese': "shibing624/text2vec-base-chinese"  # 768 dimensional
}

# Lazy loading embedding model
def get_embedding_model(language: str):
    if language not in embedding_cache:
        if language not in embedding_models:
            raise ValueError(f"Unsupported language: {language}")
        # 初始化 embedding
        embedding_cache[language] = SentenceTransformer(embedding_models[language])
    return embedding_cache[language]


# 转换为embedding向量
def embed_chunk(chunk: str, language : str= 'English') -> List[float]:
    embedding_model = get_embedding_model(language)
    embed = embedding_model.encode(chunk, normalize_embeddings=True)
    return embed.tolist()


if __name__ == '__main__':

    from chunk import split_into_sentences
    docs = "doc.md"  # 文档名
    chunk_size = 100  # 分块大小
    chunks = split_into_sentences(docs, max_chunk_size=chunk_size, language='Chinese')

    embeddings = []
    for idx, chunk in enumerate(chunks, 1):
        embeddings.append(embed_chunk(chunk, 'Chinese'))
    # print(embedding)
    print("length of embeddings: ", len(embeddings))
    print("embeddings: \n", embeddings[0])