import csv
from pymongo import MongoClient

# 🔹 Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_db"]  # Tạo database "chatbot_db"
collection = db["qa_collection"]  # Tạo collection "qa_collection"

# 🔹 Xóa dữ liệu cũ (nếu cần)
collection.delete_many({})

# 🔹 Đọc CSV và nhập vào MongoDB
with open("merged_data.csv", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = [{"question": row["Question"], "answer": row["Answer"]} for row in reader]
    collection.insert_many(data)

print("✅ Dữ liệu đã được nhập vào MongoDB!")