import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

CSV_PATH = "../database/database.csv"
QUESTION_EMBEDDINGS_PATH = "./embeddings/question_vectors.npy"
ANSWER_EMBEDDINGS_PATH = "./embeddings/answer_vectors.npy"
QUESTION_INDEX_PATH = "./embeddings/index_question.faiss"
ANSWER_INDEX_PATH = "./embeddings/index_answer.faiss"
BM25_INDEX_PATH = "./embeddings/bm25.pkl"

# Load model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Normalize function
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

# Load FAISS index
index_question = faiss.read_index(QUESTION_INDEX_PATH)
index_answer = faiss.read_index(ANSWER_INDEX_PATH)

# 🔹 Load BM25 index từ file
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_questions, bm25_answers, doc_map = pickle.load(f)

# Search in FAISS
def search_faiss(query, index, top_k=1):
    query_vector = normalize(embedding_model.encode([query]).astype('float32'))
    D, I = index.search(query_vector, top_k)
    return [(doc_map[I[0][i]]["Question"], doc_map[I[0][i]]["Answer"]) for i in range(top_k) if I[0][i] != -1]

# Search in BM25
def search_bm25_question(query, top_k=1):
    """Tìm kiếm trong danh sách câu hỏi bằng BM25"""
    if bm25_questions is None:
        return []

    tokenized_query = query.split()
    scores = bm25_questions.get_scores(tokenized_query)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [(doc_map[i]["Question"], doc_map[i]["Answer"]) for i, _ in top_n]

def search_bm25_answer(query, top_k=1):
    """Tìm kiếm trong danh sách câu trả lời bằng BM25"""
    if bm25_answers is None:
        return []

    tokenized_query = query.split()
    scores = bm25_answers.get_scores(tokenized_query)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [(doc_map[i]["Question"], doc_map[i]["Answer"]) for i, _ in top_n]

