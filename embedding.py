from sentence_transformers import SentenceTransformer
from typing import List

DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-300m"

# embedding模型全局缓存（按模型名称缓存，避免重复加载）
embedding_cache = {}
embedding_models = {
    'English': DEFAULT_EMBEDDING_MODEL,
    'Chinese': DEFAULT_EMBEDDING_MODEL,
    'Multilingual': DEFAULT_EMBEDDING_MODEL,
}


def get_embedding_model(language: str):
    """Lazy-load embedding model, defaulting to EmbeddingGemma for all languages."""
    model_name = embedding_models.get(language, DEFAULT_EMBEDDING_MODEL)
    if model_name not in embedding_cache:
        embedding_cache[model_name] = SentenceTransformer(model_name)
    return embedding_cache[model_name]


# 转换为embedding向量
def embed_chunk(chunk: str, language : str= 'Chinese') -> List[float]:
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