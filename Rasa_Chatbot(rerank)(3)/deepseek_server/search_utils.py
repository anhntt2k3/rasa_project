import chromadb
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 📂 Định nghĩa đường dẫn
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "./embeddings/bm25.pkl"

# 🛠️ Load mô hình embedding
print("🔄 Đang tải mô hình embedding...")
bi_encoder = SentenceTransformer("intfloat/e5-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# 🛠️ Kết nối ChromaDB
print("🔄 Đang kết nối ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="qa_collection")

# 🛠️ Load BM25 index từ file
print("🔄 Đang tải BM25 index...")
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_combined, doc_map = pickle.load(f)

# Hàm remove_semantic_duplicates
def remove_semantic_duplicates(results, threshold=0.9):
    """Loại bỏ các câu hỏi trùng lặp về ý nghĩa bằng cosine similarity"""
    if not results:
        return []

    questions = [q for q, _ in results]
    embeddings = bi_encoder.encode(questions)  # Mã hóa tất cả câu hỏi thành vector
    similarity_matrix = cosine_similarity(embeddings)

    unique_results = []
    seen_indices = set()

    for i in range(len(results)):
        if i in seen_indices:
            continue
        unique_results.append(results[i])
        
        # Đánh dấu những câu có độ tương đồng cao để bỏ qua
        for j in range(i + 1, len(results)):
            if similarity_matrix[i, j] > threshold:
                seen_indices.add(j)

    return unique_results

# 🔎 Tìm kiếm trong ChromaDB
def search_chromadb(query, top_k=10):
    """Tìm kiếm trong ChromaDB sử dụng Bi-Encoder"""
    query_vector = bi_encoder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)

    if not results["ids"]:
        return []
    
    return [(results["metadatas"][0][i]["Question"], results["metadatas"][0][i]["Answer"]) for i in range(len(results["ids"][0]))]


# 🔎 Tìm kiếm trong BM25
def search_bm25(query, top_k=10):
    """Tìm kiếm trong BM25 dựa vào Combined (Question + Answer)"""
    if bm25_combined is None:
        return []

    tokenized_query = query.split()
    scores = bm25_combined.get_scores(tokenized_query)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [(doc_map[i]["Question"], doc_map[i]["Answer"]) for i, _ in top_n]

# 🔥 Kết hợp BM25 + ChromaDB và rerank với Cross-Encoder
def hybrid_search(query, top_k1=10, top_k2=10, top_m=5):
    """Tìm kiếm với BM25 + ChromaDB và rerank với Cross-Encoder"""
    print(f"🔍 Query: {query}")

    # 🔹 Bước 1: Lấy kết quả từ BM25 và ChromaDB
    bm25_results = search_bm25(query, top_k1)
    chroma_results = search_chromadb(query, top_k2)

    # 🔹 Bước 2: Hợp nhất kết quả, loại bỏ trùng lặp
    combined_results = bm25_results + chroma_results
    combined_results = remove_semantic_duplicates(combined_results)

    if not combined_results:
        return ["❌ Không tìm thấy câu trả lời phù hợp."]

    # 🔹 Bước 3: Rerank bằng Cross-Encoder
    cross_scores = cross_encoder.predict([(query, f"{ans[0]} {ans[1]}") for ans in combined_results])

    # 🔹 Bước 4: Chọn ra top-M kết quả tốt nhất
    sorted_answers = [x for _, x in sorted(zip(cross_scores, combined_results), reverse=True)]
    final_answers = sorted_answers[:top_m]

    return final_answers


print("✅ Hệ thống tìm kiếm (ChromaDB + BM25) đã sẵn sàng!")
