from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from functools import lru_cache
from pymongo import MongoClient
import pandas as pd
import numpy as np
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://apnabzaar.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["GOOGLE_API_KEY"] = "AIzaSyCpCk8y8l3IU08n9_u_EWajQv-pibrBdps"


# ---------- Lazy Initialization (important for Render) ----------
products = users = orders = vector_store = None
embeddings = None
llm = None


def making_data():
    mongo_url = "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"
    client = MongoClient(mongo_url)
    db = client["ECommerce"]

    product_collection = db["products"]
    user_data_collection = db["users"]

    products = list(product_collection.find())
    users = list(user_data_collection.find())

    product_data = []
    for p in products:
        if p.get("isActive"):
            product_data.append({
                "productID": str(p["_id"]),
                "name": p["name"],
                "price": p["price"],
                "category": p["category"],
                "description": p.get("description", ""),
                "images": p.get("images", "Not Found"),
                "stock": p.get("stock", "0"),
                "rating": p.get("rating", "0"),
                "reviews": p.get("reviews", "0"),
                "createdAt": p.get("createdAt", ""),
                "updatedAt": p.get("updatedAt", ""),
                "isActive": p.get("isActive", True)
            })

    user_data = []
    order_data = []
    for u in users:
        for history in u.get("history", []):
            user_data.append({
                "user_id": str(u["_id"]),
                "productID": str(history.get("productId", "")),
                "event": history.get("event", {}).get("type", "Not Found"),
                "Timestamp": history.get("time", ""),
                "duration": history.get("duration", 0) / 1000
            })
        for order in u.get("orders", []):
            order_data.append({
                "user_id": str(u["_id"]),
                "orderID": str(order)
            })

    df_products = pd.DataFrame(product_data)
    df_user = pd.DataFrame(user_data)
    df_orders = pd.DataFrame(order_data)
    return df_products, df_user, df_orders

from threading import Thread

@app.on_event("startup")
def startup_event():
    global products, users, orders, embeddings, vector_store, llm
    print("Initializing chatbot backend...")

    products, users, orders = making_data()
    products = products[["name", "category", "price", "description"]]

    def init_vector_and_llm():
        global embeddings, vector_store, llm
        combined_text = products.to_string() + "\n" + users.to_string()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([combined_text])

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            vector_store = FAISS.load_local("faiss_index", embeddings)
            print("Loaded existing FAISS index.")
        except:
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local("faiss_index")
            print("Created and saved new FAISS index.")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        print("Chatbot ready ✅")

    # Run heavy initialization in background so port opens immediately
    Thread(target=init_vector_and_llm).start()


# ---------- Chat Logic ----------
@lru_cache(maxsize=100)
def cached_search(query):
    return vector_store.similarity_search(query, k=5)

@app.get("/chat_with_ai")
async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    try:
        docs = cached_search(question)
        context = "\n".join([d.page_content for d in docs])

        prompt = PromptTemplate(
            template="""
            You are a helpful chatbot for an e-commerce website. 
            Use ONLY the information found in the provided context. Answer concisely in 1–2 lines.
            If context lacks data, reply exactly: "No data found".

            Rules:
            1. Recommend up to 3 products using context only.
            2. If unavailable, reply with the apology message given.
            3. For order details, use only real order data.
            4. For product price, reply: "Name - Price - https://apnabzaar.netlify.app/productdetail/product_id"
            5. No extra text or assumptions.

            Context:
            {context}

            Question: {question}

            Product Data: {products}
            Order Data: {orders}
            """,
            input_variables=["context", "question", "products", "orders"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.ainvoke({
            "context": context,
            "question": question,
            "products": products.to_string(),
            "orders": orders.to_string()
        })

        return {"message": result["text"] if isinstance(result, dict) and "text" in result else str(result)}

    except Exception as e:
        print("Error in chat_ai_async:", repr(e))
        return {"message": f"Internal error: {str(e)}"}


# ---------- API Endpoints ----------
@app.get("/chat")
async def chat(user_id: str, option: str):
    if option == "main":
        return JSONResponse({
            "message": "Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
            "options": ["Order Related", "Product Related", "Others"]
        })
    elif option == "Order Related":
        return JSONResponse({"message": "Please choose an option related to your orders:",
                             "options": ["Recent Order", "All Orders", "Track Order", "Back"]})
    elif option == "Product Related":
        return JSONResponse({"message": "Need help with products? Select an option below:",
                             "options": ["Request Product", "Back"]})
    elif option == "Others":
        return JSONResponse({"message": "You can chat with our AI assistant for general help 💬",
                             "options": ["Chat with AI Assistant", "Back"]})
    elif option == "Recent Order":
        user_id_orders = orders[orders["user_id"] == user_id]
        return user_id_orders[-1:]
    elif option == "All Orders":
        user_id_orders = orders[orders["user_id"] == user_id]
        return user_id_orders[-5:]
    elif option == "Track Order":
        user_id_orders = orders[orders["user_id"] == user_id]
        return user_id_orders[-1:]
    elif option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).",
                             "options": ["Back"]})
    elif option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:",
                             "options": ["Back"]})
    elif option == "Back":
        return JSONResponse(await chat(user_id, "main"))

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})

# ---------- Render Entry Point ----------
if __name__ == "__main__":
    import sys
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))  # Render auto-assigns port
    print(f"🚀 Starting server on port {port}", file=sys.stderr)
    uvicorn.run("chatbot:app", host="0.0.0.0", port=port)

