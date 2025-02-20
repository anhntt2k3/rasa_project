import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from deep_translator import GoogleTranslator
from search_utils import search_faiss, search_bm25_answer, search_bm25_question, index_question, index_answer #Import search_faiss and search_bm25 functions from search_utils.py
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langdetect import detect
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

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
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    max_len=5
)

#Extract keywords
def extract_keywords(user_question):
    doc = nlp_en(user_question)
    keywords = [chunk.text for chunk in doc.noun_chunks][:3]
    return keywords

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

    #Search the FAISS index
    faiss_results = search_faiss(user_question, index_question) + search_faiss(user_question, index_answer)
    print(f"FAISS results: {faiss_results}") #Debug

    #Search the BM25 index
    bm25_results = []
    for kw in keywords:
        bm25_results += search_bm25_question(kw, top_k=1) + search_bm25_answer(kw, top_k=1)
    print(f"BM25 results: {bm25_results}") #Debug

    #Combine the results
    combined_results = list({(q, a) for q, a in faiss_results + bm25_results})
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
        - If information is insufficient, ask for more details.
        - Maintain conversation context.
        - Answer only within the MOR Software database.
        - Always respond in the language of the question.
        - Provide clear and concise responses.
        - **Do NOT** include "Think" or "Reasoning" sections.
        - If the question is **irrelevant** (e.g., personal topics like food, weather, or unrelated subjects), **do not answer**. Instead, respond with:  "I can only assist with questions related to MOR Software. Please ask a relevant question."
        - If the question is **related to MOR Software** but **not found in the database**, suggest contacting MOR Software directly: "This information is not available in our system. Please contact MOR Software directly for further details."
        
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
        return {"answer": response}
    
    except Exception as e:
        return {"answer": f"Error in LangChain call: {str(e)}"}
