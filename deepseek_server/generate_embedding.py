import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os
import pickle

# Paths
CSV_PATH = "../database/database.csv"
QUESTION_EMBEDDINGS_PATH = "./embeddings/question_vectors.npy"
ANSWER_EMBEDDINGS_PATH = "./embeddings/answer_vectors.npy"
QUESTION_INDEX_PATH = "./embeddings/index_question.faiss"
ANSWER_INDEX_PATH = "./embeddings/index_answer.faiss"
BM25_INDEX_PATH = "./embeddings/bm25.pkl"
# Load dataset
print("🔄 Đang tải database...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")[['Question', 'Answer']]
df = df.apply(lambda x: x.str.lower().str.strip())

# Load Sentence Transformer model
print("🔄 Đang tải mô hình embedding...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Normalize function
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

# Create embeddings for questions and answers
print("🔄 Đang tạo embedding cho câu hỏi và câu trả lời...")
question_vectors = normalize(embedding_model.encode(df["Question"].tolist()).astype('float32'))
answer_vectors = normalize(embedding_model.encode(df["Answer"].tolist()).astype('float32'))

# Save embeddings to file
print("💾 Đang lưu embeddings vào file...")
np.save(QUESTION_EMBEDDINGS_PATH, question_vectors)
np.save(ANSWER_EMBEDDINGS_PATH, answer_vectors)

# Create FAISS index
print("🔄 Đang tạo FAISS index...")
index_question = faiss.IndexHNSWFlat(question_vectors.shape[1], 32)
index_question.add(question_vectors)
index_answer = faiss.IndexHNSWFlat(answer_vectors.shape[1], 32)
index_answer.add(answer_vectors)

# Save FAISS index
print("💾 Đang lưu FAISS index...")
faiss.write_index(index_question, QUESTION_INDEX_PATH)
faiss.write_index(index_answer, ANSWER_INDEX_PATH)

# Tokenize questions and answers
print("🔄 Đang tạo BM25 index...")
tokenized_questions = [q.split() for q in df['Question'].tolist()]
tokenized_answers = [a.split() for a in df['Answer'].tolist()]

# Create BM25 index
bm25_questions = BM25Okapi(tokenized_questions)
bm25_answers = BM25Okapi(tokenized_answers)

# Save BM25 index
with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump((bm25_questions, bm25_answers, df.to_dict(orient='records')), f)

print("✅ Embeddings, FAISS index và BM25 đã được tạo & lưu!")
