import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

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
    query_vector = bi_encoder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)

    if not results["ids"]:
        return []
    
    return [(results["metadatas"][0][i]["question"], results["metadatas"][0][i]["answer"]) for i in range(len(results["ids"][0]))]

# 🔥 Tìm kiếm và rerank với Cross-Encoder
def search_with_rerank(query, top_k=10, top_m=3):
    """Tìm kiếm với ChromaDB và rerank với Cross-Encoder"""
    print(f"🔍 Query: {query}")

    # 🔹 Bước 1: Lấy kết quả từ ChromaDB
    chroma_results = search_chromadb(query, top_k)
    chroma_results = remove_semantic_duplicates(chroma_results)

    if not chroma_results:
        return ["❌ Không tìm thấy câu trả lời phù hợp."]

    # 🔹 Bước 2: Rerank bằng Cross-Encoder
    cross_scores = cross_encoder.predict([(query, f"{ans[0]} {ans[1]}") for ans in chroma_results])

    # 🔹 Bước 3: Chọn ra top-M kết quả tốt nhất
    sorted_answers = [x for _, x in sorted(zip(cross_scores, chroma_results), reverse=True)]
    final_answers = sorted_answers[:top_m]

    return final_answers

print("✅ Hệ thống tìm kiếm (ChromaDB) đã sẵn sàng!")
