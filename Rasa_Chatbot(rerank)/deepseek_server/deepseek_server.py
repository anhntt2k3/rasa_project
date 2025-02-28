import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from deep_translator import GoogleTranslator
# from translation import translate
from search_utils import search_chromadb, search_bm25, hybrid_search # Import search functions
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langdetect import detect
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

app = FastAPI()

# Spacy model for English
nlp_en = spacy.load("en_core_web_sm")

#API key for the deepseek model
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

#Database path
csv_path = "../database/database.csv"
df = pd.read_csv(csv_path, encoding="utf-8")[['Question', 'Answer']]
df = df.apply(lambda x: x.str.lower().str.strip())


#Conversation buffer memory
memory = ConversationBufferWindowMemory(
    k=5,  # Chỉ lưu 5 tin nhắn gần nhất
    memory_key="history",
    return_messages=True
)

#Extract keywords
def extract_keywords(user_question, num_keywords=3):
    doc = nlp_en(user_question)

    # Chỉ chọn danh từ, động từ, tính từ quan trọng
    keywords = [
        token.text for token in doc 
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop
    ]
    
    return keywords[:num_keywords]  # Lấy tối đa num_keywords


class QuestionRequest(BaseModel):
    question: str

#API endpoint for querying the deepseek model
@app.post("/query")
def query_deepseek(request: QuestionRequest):
    user_question = request.question.lower().strip()
    lang = detect(user_question)
    flag_translate = False
    if lang == "vi":
        user_question = GoogleTranslator(source='vi', target='en').translate(user_question)
        flag_translate = True

    keywords = extract_keywords(user_question)
    print(f"User question: {user_question}") #Debug
    print(f"Keywords: {keywords}")  #Debug

    # 🔍 Tìm kiếm với hybrid_search
    hybrid_results = hybrid_search(user_question, top_k1=10, top_k2=10, top_m=3)
    print(f"Hybrid Search results: {hybrid_results}")  # Debug

    if not hybrid_results:
        return {"answer": "❌ Không tìm thấy câu trả lời phù hợp."}

    #Create the context from the database
    context = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in hybrid_results])
    print(f"Context: {context}") #Debug

    #Prompt for DeepSeek api
    prompt = PromptTemplate(
        template=""" 

        Relevant Data:
        {context}

        You are a customer support assistant for MOR Software.  
        Your role is to assist users with accurate and professional responses based on the MOR Software database.  

        ### General Instructions:  
        - If the question **lacks details**, ask follow-up questions to clarify.  
        - If the question is **clear** but not in the database, do **not generate answers**. Instead, reply:  
        **"This information is not available in our system. Please contact MOR Software directly for further details."**  
        - Maintain conversation context and keep track of previous interactions.  
        - If the conversation shifts to an unrelated topic, gently guide the user back to MOR Software-related topics.  
        - Answer **only within the MOR Software database**.  

        ### Language & Formatting:  
        - Always respond in the **same language as the question**.  
        - Provide **clear and concise** responses.  
        - **Do NOT** include "Think" or "Reasoning" sections.  
        - Format responses in a structured way (e.g., bullet points, short paragraphs).  
        - Summarize long responses while keeping key details intact.  
        - Maintain a **professional yet friendly tone**.  

        ### Handling Different Types of Questions:  

        #### 1️⃣ **Important Company Information (e.g., MOR's address, Board of Directors, official policies)**  
        - Ensure responses are **100% accurate**, sourced from the database.  
        - If the information is not available, **do not guess**. Instead, respond:  
        **"This information is not available in our system. Please contact MOR Software directly for further details."**  

        #### 2️⃣ **Company-Related Questions (e.g., services, technologies, workflows, hiring process)**  
        - If the database has relevant information, provide a **structured answer**.  
        - If the question is reasonable but not covered in the database, provide a **general response** based on MOR’s expertise, while ensuring it remains truthful. Example:  
        **"MOR Software specializes in a wide range of technologies, including [relevant examples]. Please contact our team for more details."**  

        #### 3️⃣ **Unrelated or Off-Topic Questions**  
        - If the question is unrelated to MOR (e.g., "What's the weather today?" or "What should I eat for lunch?"), politely refuse to answer:  
        **"I can only assist with questions related to MOR Software. Please ask a relevant question."**  
        - If the user insists on unrelated topics, guide them back to MOR-related discussions.  

        ### Example Scenarios:  

        **User:** "What services does MOR Software provide?"  
        **Response:**  
        ✅ "MOR Software offers a variety of services, including:  
        - Custom software development  
        - Web and mobile app development  
        - AI and data analytics solutions  
        For more details, visit our official website or contact our support team."  

        **User:** "Where is MOR Software’s headquarters?"  
        **Response (if found in database):**  
        ✅ "MOR Software’s headquarters is located at [official address]."  
        **Response (if not in database):**  
        ✅ "This information is not available in our system. Please contact MOR Software directly for further details."  

        **User:** "What’s the weather like today?"  
        **Response:**  
        🚫 "I can only assist with questions related to MOR Software. Please ask a relevant question."  

        ---        
        
        Conversation History:
        {history}

        Current Question:
        {user_question}
        """,
        input_variables=["user_question", "context", "history"]
    )

    print("Prompt expected input variables:", prompt.input_variables) #Debug

    #Call the DeepSeek model
    llm = ChatOpenAI(
        openai_api_key=deepseek_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1-distill-llama-70b:free",
        temperature=0.5,
        max_tokens=2048,
        # extra_body={
        #     "think": False  # OpenRouter có hỗ trợ param này không? Nếu không, hãy bỏ nó đi.
        # }
    )

    #LangChain pipeline
    chain = (
        {
            "user_question": RunnableLambda(lambda x: x["user_question"]), 
            "context": RunnableLambda(lambda x: x["context"]), 
            "history": RunnableLambda(lambda _: "\n".join([msg.content for msg in memory.load_memory_variables({})["history"]]) or "No history")
        } 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    #Invoke the LangChain pipeline
    try:
        response = chain.invoke({
            "user_question": user_question,
            "context": context,
            # "history": "\n".join([msg.content for msg in memory.chat_memory.messages]) or "No history"
        })
        memory.save_context(
            {"input": user_question},
            {"output": response}
        )
        if flag_translate:
            response = GoogleTranslator(source='en', target='vi').translate(response)

        print("Conversation history:", memory.load_memory_variables({})["history"]) #Debug

        # 🛠 GỌI API ĐÁNH GIÁ TỪ `evaluate_chatbot_ragas.py`
        eval_payload = {
            "question": user_question,
            "contexts": [context],  # Truyền ngữ cảnh tìm thấy từ FAISS/BM25
            "answer": response
        }

        eval_url = "http://localhost:8001/evaluate"  # API đánh giá chạy trên cổng 8001
        try:
            eval_result = requests.post(eval_url, json=eval_payload).json()
            print("Evaluation Result:", eval_result)  # Debug kết quả đánh giá
        except Exception as e:
            eval_result = {"error": str(e)}
            print("Evaluation Error:", e)

        # 📌 Trả về câu trả lời cùng điểm đánh giá
        return {
            "answer": response,
            "evaluation": eval_result  # Thêm kết quả đánh giá vào phản hồi
        }

    
    except Exception as e:
        return {"answer": f"Error in LangChain call: {str(e)}"}
