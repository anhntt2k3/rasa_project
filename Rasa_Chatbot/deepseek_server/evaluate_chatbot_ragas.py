import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load API key từ file .env
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Kiểm tra API key
if not DEEPSEEK_API_KEY:
    raise ValueError("🚨 Lỗi: Chưa có API key. Hãy đặt DEEPSEEK_API_KEY trong file .env!")

# Khởi tạo mô hình DeepSeek
llm = ChatOpenAI(
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat:free",
    temperature=0.5,
    max_tokens=1024,
)

# Định nghĩa dữ liệu đầu vào
class EvalRequest(BaseModel):
    question: str
    contexts: list
    answer: str

@app.post("/evaluate")
def evaluate_response(request: EvalRequest):
    """
    API đánh giá chatbot dựa trên DeepSeek:
    - Độ chính xác (Faithfulness)
    - Mức độ liên quan (Context Precision)
    - Độ đúng của câu trả lời (Answer Correctness)
    """
    prompt = f"""
        You are an expert in evaluating chatbots. Please assess the response based on the following criteria:  
        1️⃣ **Faithfulness**: Does the answer rely on the information provided in the contexts?  
        2️⃣ **Context Precision**: Does the answer align with the question?  
        3️⃣ **Answer Correctness**: Is the answer correct?  

        Data:  
        - **Question**: {request.question}  
        - **Contexts**: {request.contexts}  
        - **Answer**: {request.answer}  

        **No explanation is needed**, just provide ratings from 0 to 1 (0: lowest, 1: highest).  
        Return the evaluation results in the following JSON format:  
        {{  
            "faithfulness": provide a decimal number without rounding from 0 to 1 (e.g., 0.99233),  
            "context_precision": 0-1,  
            "answer_correctness": 0-1  
        }}

    """

    # Gửi request đến DeepSeek
    response = llm.invoke(prompt)

    #debug
    print("Question: ",request.question)
    print("Contexts: ",request.contexts)
    print("Answer: ",request.answer)

    # Kiểm tra response có ở dạng JSON không
    try:
        json_str = response.content.strip("```json\n").strip("\n```")  # Loại bỏ markdown JSON
        json_data = json.loads(json_str)  # Chuyển về Python dict
    except Exception as e:
        print("🚨 Lỗi xử lý JSON:", e)
        json_data = {"error": "Phản hồi từ mô hình không phải JSON hợp lệ.", "raw_response": response.content}

    print("📊 Kết quả đánh giá JSON:", json_data)  # Debug

    return json_data  # Trả về JSON chuẩn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
