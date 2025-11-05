from chunk import split_into_sentences
from embedding import embed_chunk
from DB import save_embeddings, get_db

docs = "doc.md"
chunks = split_into_sentences(docs, max_chunk_size=100, language="Chinese")
embeddings = [embed_chunk(chunk, "Chinese") for chunk in chunks]

# persistent=True 本地建库
db = get_db(persistent=True, name="default")
save_embeddings(db, chunks, embeddings)

print("Database built. Count:", db.count())
