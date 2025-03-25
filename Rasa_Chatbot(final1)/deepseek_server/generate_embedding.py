import time
import pandas as pd
import chromadb
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Paths
# BM25_INDEX_PATH = "./embeddings/bm25.pkl"
CHROMA_DB_PATH = "./chroma_db"

# 🛠️ Kết nối MongoDB
print("🔄 Đang tải dữ liệu từ MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_db"]
collection = db["qa_collection"]

# 🛠️ Lấy dữ liệu từ MongoDB
data = list(collection.find({}, {"_id": 0, "question": 1, "answer": 1}))  
df = pd.DataFrame(data)

# 🛠️ Xử lý dữ liệu
df = df.apply(lambda x: x.str.lower().str.strip())
print(f"✅ Đã tải {len(df)} câu hỏi từ MongoDB!")

# 🛠️ Load mô hình embedding
print("🔄 Đang tải mô hình embedding...")
embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# 🛠️ Ghép câu hỏi và câu trả lời
df["Combined"] = df["question"] + " " + df["answer"]

# 🛠️ Tạo embeddings
start_time = time.time()
print("🔄 Đang tạo embedding cho cả câu hỏi và câu trả lời...")
combined_vectors = embedding_model.encode(df["Combined"].tolist()).tolist()
print(f"✅ Tạo embeddings xong! ⏱️ {time.time() - start_time:.2f}s")

# 🛠️ Kết nối ChromaDB
start_time = time.time()
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)  
collection = chroma_client.get_or_create_collection(name="qa_collection")

# 🛠️ Thêm dữ liệu vào ChromaDB
print("🔄 Đang thêm dữ liệu vào ChromaDB...")
for i, row in df.iterrows():
    collection.add(
        ids=[str(i)],  
        embeddings=[combined_vectors[i]],  
        metadatas=[{"question": row["question"], "answer": row["answer"]}]
    )
print(f"✅ Dữ liệu đã được lưu vào ChromaDB! ⏱️ {time.time() - start_time:.2f}s")

# # 🛠️ Tạo BM25 index trên cả question + answer
# start_time = time.time()
# print("🔄 Đang tạo BM25 index trên cả question và answer...")
# tokenized_texts = [text.split() for text in df['Combined'].tolist()]
# bm25 = BM25Okapi(tokenized_texts)
# print(f"✅ Tạo BM25 index xong! ⏱️ {time.time() - start_time:.2f}s")

# # 🛠️ Lưu BM25 index
# with open(BM25_INDEX_PATH, "wb") as f:
#     pickle.dump((bm25, df.to_dict(orient='records')), f)

# print("✅ ChromaDB và BM25 đã được tạo & lưu!")
