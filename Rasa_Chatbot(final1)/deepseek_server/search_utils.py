import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time

# 📂 Định nghĩa đường dẫn
CHROMA_DB_PATH = "./chroma_db"

# 🛠️ Load mô hình embedding
print("🔄 Đang tải mô hình embedding...")
bi_encoder = SentenceTransformer("intfloat/e5-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# 🛠️ Kết nối ChromaDB
print("🔄 Đang kết nối ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="qa_collection")

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
    start_embed_time = time.time()  # ⏱ Bắt đầu đo thời gian embedding

    query_vector = bi_encoder.encode([query]).tolist()
    embed_time = time.time() - start_embed_time  # ⏱ Tính thời gian embedding
    print(f"⚡ Embedding time: {embed_time:.4f} seconds")  # Debug log

    start_search_time = time.time()  # ⏱ Bắt đầu đo thời gian search
    results = collection.query(query_embeddings=query_vector, n_results=top_k)
    search_time = time.time() - start_search_time  # ⏱ Tính thời gian search
    print(f"🔎 ChromaDB search time: {search_time:.4f} seconds")  # Debug log

    if not results["ids"]:
        return []
    
    return [(results["metadatas"][0][i]["question"], results["metadatas"][0][i]["answer"]) for i in range(len(results["ids"][0]))]

# 🔥 Tìm kiếm và rerank với Cross-Encoder
def rerank_results(query, search_results, top_m=3):
    """Rerank danh sách kết quả tìm kiếm với Cross-Encoder"""

    start_time = time.time()  # ⏱ Bắt đầu đo thời gian

    if not search_results:
        return ["❌ Không tìm thấy câu trả lời phù hợp."]
    
    search_results = remove_semantic_duplicates(search_results, threshold=0.9)

    # Tạo danh sách cặp (query, question + answer)
    cross_encoder_inputs = [(query, f"{qa[0]} {qa[1]}") for qa in search_results]

    # Tính điểm tương đồng
    cross_scores = cross_encoder.predict(cross_encoder_inputs)

    # Sắp xếp theo điểm số giảm dần
    sorted_results = sorted(zip(search_results, cross_scores), key=lambda x: x[1], reverse=True)

    # Chỉ lấy top-M kết quả mà không cần kiểm tra ngưỡng
    final_answers = [q_a for (q_a, _) in sorted_results[:top_m]]

    elapsed_time = time.time() - start_time  # ⏱ Tính thời gian thực thi
    print(f"⚡ Rerank time: {elapsed_time:.4f} seconds")  # Debug log

    return final_answers

print("✅ Hệ thống tìm kiếm (ChromaDB) đã sẵn sàng!")
