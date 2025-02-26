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
from search_utils import search_chromadb, search_bm25 # Import search functions
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

    # 🔍 Tìm kiếm trên ChromaDB
    chromadb_results = search_chromadb(user_question, top_k=3)
    print(f"ChromaDB results: {chromadb_results}") #Debug

    # 🔍 Tìm kiếm trên BM25 (tìm theo từ khóa)
    bm25_results = search_bm25(user_question, top_k=1)  # Tìm kiếm theo từ khóa
    print(f"BM25 results: {bm25_results}")  # Debug

    #Combine the results
    combined_results = list({(q, a) for q, a in chromadb_results + bm25_results})
    if not combined_results:
        return {"answer": "No relevant answer found in the database."}

    #Create the context from the database
    context = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in combined_results])
    print(f"Context: {context}") #Debug

    #Prompt for DeepSeek api
    prompt = PromptTemplate(
        template=""" 

        Relevant Data:
        {context}

        You are a customer support assistant for MOR Software.  
        - If the question **lacks details**, ask follow-up questions to clarify.  
        - If the question is **clear** but not in the database, do **not generate answers**. Instead, reply:  
        "This information is not available in our system. Please contact MOR Software directly for further details."  
        - Maintain conversation context and keep track of previous interactions.  
        - If the conversation shifts to an unrelated topic, gently guide the user back to MOR Software-related topics.  
        - Answer **only within the MOR Software database**.  
        - Always respond in the **same language as the question**.  
        - Provide **clear and concise** responses.  
        - **Do NOT** include "Think" or "Reasoning" sections.  
        - Format responses in a structured way (e.g., bullet points, short paragraphs).  
        - Summarize long responses while keeping key details intact.  
        - Maintain a **professional yet friendly tone**.  
        - **If the question is irrelevant** (e.g., personal topics like food, weather, or unrelated subjects), **do not answer**. Instead, respond with:  
        "I can only assist with questions related to MOR Software. Please ask a relevant question."          
        
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
        max_tokens=1024,
        # extra_body={
        #     "think": False  # OpenRouter có hỗ trợ param này không? Nếu không, hãy bỏ nó đi.
        # }
    )

    #LangChain pipeline
    chain = (
        {"user_question": RunnableLambda(lambda x: x), 
        "context": RunnableLambda(lambda x: x), 
        "history": RunnableLambda(lambda x: x)} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    #Invoke the LangChain pipeline
    try:
        response = chain.invoke({
            "user_question": user_question,
            "context": context,
            "history": "\n".join([msg.content for msg in memory.chat_memory.messages]) or "No history"
        })
        memory.save_context(
            {"input": user_question},
            {"output": response}
        )
        if flag_translate:
            response = GoogleTranslator(source='en', target='vi').translate(response)

        print("Conversation history:", memory.chat_memory.messages) #Debug

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
