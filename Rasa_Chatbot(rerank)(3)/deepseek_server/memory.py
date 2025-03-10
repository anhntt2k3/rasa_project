# import chromadb
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Khởi tạo mô hình nhúng văn bản (intfloat/e5-base-v2)
# embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# # Khởi tạo ChromaDB
# chroma_client = chromadb.PersistentClient(path="./chroma_db_memory")  # Lưu trữ trên ổ đĩa
# collection = chroma_client.get_or_create_collection(name="chat_memory")

# import json

# def save_history(user_question, bot_response):
#     """Lưu cả question_embedding và answer_embedding vào metadata dưới dạng chuỗi JSON."""
#     question_embedding = embedding_model.encode(user_question).tolist()
#     answer_embedding = embedding_model.encode(bot_response).tolist()

#     existing_ids = collection.get()["ids"]
#     new_id = str(len(existing_ids)) if existing_ids else "0"

#     collection.add(
#         ids=[new_id],
#         documents=[f"User: {user_question}\nBot: {bot_response}"],
#         embeddings=[question_embedding],  # ✅ Chỉ lưu question_embedding vào ChromaDB
#         metadatas=[{
#             "question": user_question,
#             "answer": bot_response,
#             "question_embedding": json.dumps(question_embedding),  # ✅ Lưu dưới dạng JSON string
#             "answer_embedding": json.dumps(answer_embedding)  # ✅ Lưu dưới dạng JSON string
#         }]
#     )

#     print(f"[DEBUG] Lưu lịch sử: {new_id} -> {user_question} | {bot_response}")  # Debug



# def get_relevant_history(user_question, top_k=3, threshold=0.5):
#     """Truy xuất lịch sử dựa trên độ tương đồng cao nhất giữa câu hỏi & câu trả lời."""
#     all_docs = collection.get(include=["metadatas"])

#     if not all_docs["metadatas"]:
#         return "No history"

#     # Mã hóa câu hỏi hiện tại
#     search_embedding = np.array(embedding_model.encode(user_question))

#     question_similarities = []
#     answer_similarities = []

#     # Duyệt qua từng mục trong metadata để so sánh riêng biệt
#     for meta in all_docs["metadatas"]:
#         question_embedding = np.array(json.loads(meta["question_embedding"]))  # ✅ Giải mã JSON
#         answer_embedding = np.array(json.loads(meta["answer_embedding"]))  # ✅ Giải mã JSON

#         # Tính độ tương đồng cosine
#         sim_q = np.dot(question_embedding, search_embedding)
#         sim_a = np.dot(answer_embedding, search_embedding)

#         question_similarities.append(sim_q)
#         answer_similarities.append(sim_a)

#     question_similarities = np.array(question_similarities)
#     answer_similarities = np.array(answer_similarities)

#     # Chọn độ tương đồng lớn nhất giữa câu hỏi và câu trả lời
#     similarities = np.maximum(question_similarities, answer_similarities)

#     # Lấy top-k có điểm cao nhất
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     top_scores = similarities[top_indices]

#     # Nếu không có đoạn nào đạt ngưỡng, lấy k đoạn gần nhất theo thời gian
#     if max(top_scores) < threshold:
#         top_indices = list(range(max(len(all_docs["metadatas"]) - top_k, 0), len(all_docs["metadatas"])))

#     relevant_history = []
#     for i in top_indices:
#         metadata = all_docs["metadatas"][i]
#         question = metadata.get("question", "")
#         answer = metadata.get("answer", "")
#         relevant_history.append(f"User: {question}\nBot: {answer}")

#     return "\n".join(relevant_history) if relevant_history else "No history"



# def reset_memory():
#     """Reset ChromaDB bằng cách xóa toàn bộ dữ liệu trong collection"""
#     global collection
#     all_ids = collection.get(include=["documents"])["ids"]  # Lấy danh sách ID
#     if all_ids:
#         collection.delete(ids=all_ids)  # Xóa tất cả ID đã lấy
#         print("[INFO] ChromaDB đã được reset khi khởi chạy")
#     else:
#         print("[INFO] Không có dữ liệu để xóa trong ChromaDB")
