import chromadb
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# 📂 Định nghĩa đường dẫn
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "./embeddings/bm25.pkl"

# 🛠️ Load mô hình embedding
print("🔄 Đang tải mô hình embedding...")
embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# 🛠️ Kết nối ChromaDB
print("🔄 Đang kết nối ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="qa_collection")

# 🛠️ Load BM25 index từ file
print("🔄 Đang tải BM25 index...")
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_combined, doc_map = pickle.load(f)

# 🔎 Tìm kiếm trong ChromaDB
def search_chromadb(query, top_k=1):
    """Tìm kiếm trong ChromaDB sử dụng embedding"""
    query_vector = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)

    if not results["ids"]:
        return []
    
    return [(results["metadatas"][0][i]["Question"], results["metadatas"][0][i]["Answer"]) for i in range(len(results["ids"][0]))]

# 🔎 Tìm kiếm trong BM25
def search_bm25(query, top_k=1):
    """Tìm kiếm trong BM25 dựa vào Combined (Question + Answer)"""
    if bm25_combined is None:
        return []

    tokenized_query = query.split()
    scores = bm25_combined.get_scores(tokenized_query)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [(doc_map[i]["Question"], doc_map[i]["Answer"]) for i, _ in top_n]

print("✅ Hệ thống tìm kiếm (ChromaDB + BM25) đã sẵn sàng!")
