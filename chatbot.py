from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import BaseModel
import pandas as pd
from pymongo import MongoClient
import numpy as np
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://localhost:5174',
        'https://apnabzaar.netlify.app'
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment variables for sensitive data
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyAIzbhiQ1Ga-XfzozyoYugrrhwAXtdrxB8')
MONGO_URL = os.getenv('MONGO_URL', 'mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def making_data():
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        
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
                    'user_id': str(u["_id"]),
                    "orderID": order
                })

        df_products = pd.DataFrame(product_data)
        df_user = pd.DataFrame(user_data)
        df_orders = pd.DataFrame(order_data)

        client.close()
        return df_products, df_user, df_orders
    
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        # Return empty DataFrames if connection fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Global variables
products = pd.DataFrame()
users = pd.DataFrame()
orders = pd.DataFrame()
vector_store = None
llm = None

@app.on_event("startup")
async def startup_event():
    global products, users, orders, vector_store, llm
    
    try:
        logger.info("Loading data from MongoDB...")
        products, users, orders = making_data()
        
        if not products.empty:
            products = products[['name', 'category', 'price', 'description', 'productID']]
            
            logger.info("Creating embeddings...")
            combined_text = products.to_string() + "\n" + users.to_string()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([combined_text])

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            try:
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.info(f"Creating new FAISS index: {str(e)}")
                vector_store = FAISS.from_documents(chunks, embeddings)
                vector_store.save_local("faiss_index")

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5)
            logger.info("Startup complete!")
        else:
            logger.warning("No products loaded from database")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@lru_cache(maxsize=100)
def cached_search(query: str):
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=5)

async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    if vector_store is None or llm is None:
        return {"message": "AI service is initializing. Please try again in a moment."}

    try:
        docs = cached_search(question)
        context = "\n".join([d.page_content for d in docs])

        prompt = PromptTemplate(
            template="""
            You are a helpful chatbot for an e-commerce website. 
            Use ONLY the information found in the provided context. Answer concisely in 1–2 lines.

            If the context does not contain enough information to answer, reply exactly: "No data found".

            Rules:
            1. If the user requests a product recommendation, recommend up to 3 products that best match the user's preferences and needs, using only product fields present in the context.
            2. If the user asks for a product that is not available in the context, reply exactly:
            "We are sorry, the product you requested is currently not available on our site. However, we value your interest and would be happy to assist you with similar products or alternatives that meet your needs. Please let us know what you're looking for, and we'll do our best to help you find a suitable option."
            3. If the user asks for order details, return only order information present in the order data (e.g., order id, items, status, delivery ETA). Do NOT invent or assume missing fields.
            4. If the user asks for a product's price, reply with the price, Name of product and the product link in this exact format:
            https://apnabzaar.netlify.app/productdetail/product_id
            Replace `product_id` with the product's `id` value from the Product Data.
            5. Do not provide any information that is not present in the context. Do not add technical notes, disclaimers, or extra sentences—keep it to 1–2 lines.

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
            "products": products.to_string() if not products.empty else "No products available", 
            "orders": orders.to_string() if not orders.empty else "No orders available"
        })

        return {"message": result["text"] if isinstance(result, dict) and "text" in result else str(result)}

    except Exception as e:
        logger.error(f"Error in chat_ai_async: {repr(e)}")
        return {"message": "Sorry, I encountered an error. Please try again."}

@app.get("/")
async def root():
    return {"message": "ApnaBazzar API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "products_loaded": not products.empty,
        "vector_store_ready": vector_store is not None,
        "llm_ready": llm is not None
    }

@app.get("/chat")
async def chat(user_id: str, option: str):
    if option == "main":
        return JSONResponse({
            "message": "Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
            "options": ["Order Related", "Product Related", "Others"]
        })
    elif option == "Order Related":
        return JSONResponse({
            "message": "Please choose an option related to your orders:",
            "options": ["Recent Order", "All Orders", "Track Order", "Back"]
        })
    elif option == "Product Related":
        return JSONResponse({
            "message": "Need help with products? Select an option below:",
            "options": ["Request Product", "Back"]
        })
    elif option == "Others":
        return JSONResponse({
            "message": "You can chat with our AI assistant for general help 💬",
            "options": ["Chat with AI Assistant", "Back"]
        })

    elif option == "Recent Order":
        user_id_orders = orders[orders['user_id'] == user_id]
        if user_id_orders.empty:
            return JSONResponse({"message": "No orders found", "options": ["Back"]})
        return JSONResponse({"orders": user_id_orders.tail(1).to_dict('records'), "options": ["Back"]})

    elif option == "All Orders":
        user_id_orders = orders[orders['user_id'] == user_id]
        if user_id_orders.empty:
            return JSONResponse({"message": "No orders found", "options": ["Back"]})
        return JSONResponse({"orders": user_id_orders.tail(5).to_dict('records'), "options": ["Back"]})

    elif option == "Track Order":
        user_id_orders = orders[orders['user_id'] == user_id]
        if user_id_orders.empty:
            return JSONResponse({"message": "No orders found", "options": ["Back"]})
        return JSONResponse({"orders": user_id_orders.tail(1).to_dict('records'), "options": ["Back"]})

    elif option == "Request Product":
        return JSONResponse({
            "message": "Send us the product name you want to request (not available on site).",
            "options": ["Back"]
        })

    elif option == "Chat with AI Assistant":
        return JSONResponse({
            "message": "You're now connected to the AI Assistant. Please type your question below:",
            "options": ["Back"]
        })

    elif option == "Back":
        return await chat(user_id, "main")

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})

@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chatbot:app", host="0.0.0.0", port=port, reload=False)
