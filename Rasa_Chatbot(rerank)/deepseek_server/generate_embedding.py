import time
import pandas as pd
import chromadb
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Paths
CSV_PATH = "../database/database.csv"
BM25_INDEX_PATH = "./embeddings/bm25.pkl"
CHROMA_DB_PATH = "./chroma_db"

# 🛠️ Load database
print("🔄 Đang tải database...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")[['Question', 'Answer']]
df = df.apply(lambda x: x.str.lower().str.strip())

# 🛠️ Load mô hình embedding
print("🔄 Đang tải mô hình embedding...")
embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# 🛠️ Ghép Question và Answer trước khi tạo embedding
df["Combined"] = df["Question"] + " " + df["Answer"]

# 🛠️ Tạo embeddings
start_time = time.time()
print("🔄 Đang tạo embedding cho cả câu hỏi và câu trả lời...")
combined_vectors = embedding_model.encode(df["Combined"].tolist()).tolist() #shape: (n_samples, 768)
print(f"✅ Tạo embeddings xong! ⏱️ {time.time() - start_time:.2f}s")

# 🛠️ Kết nối ChromaDB (Lưu trên ổ đĩa)
start_time = time.time()
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)  
collection = chroma_client.get_or_create_collection(name="qa_collection")

# 🛠️ Thêm dữ liệu vào ChromaDB
print("🔄 Đang thêm dữ liệu vào ChromaDB...")
for i, row in df.iterrows():
    collection.add(
        ids=[str(i)],  
        embeddings=[combined_vectors[i]],  
        metadatas=[{"Question": row["Question"], "Answer": row["Answer"]}]
    )
print(f"✅ Dữ liệu đã được lưu vào ChromaDB! ⏱️ {time.time() - start_time:.2f}s")

# 🛠️ Tạo BM25 index trên cả Question + Answer
start_time = time.time()
print("🔄 Đang tạo BM25 index trên cả Question và Answer...")
tokenized_texts = [text.split() for text in df['Combined'].tolist()]
bm25 = BM25Okapi(tokenized_texts)
print(f"✅ Tạo BM25 index xong! ⏱️ {time.time() - start_time:.2f}s")

# 🛠️ Lưu BM25 index
with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump((bm25, df.to_dict(orient='records')), f)

print("✅ ChromaDB và BM25 đã được tạo & lưu!")
