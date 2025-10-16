from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from functools import lru_cache
from pymongo import MongoClient
import pandas as pd
import os

app = FastAPI()

# ----------------- Middleware -----------------
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

# ----------------- Environment -----------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyAIzbhiQ1Ga-XfzozyoYugrrhwAXtdrxB8")

# ----------------- MongoDB Data -----------------
def making_data():
    mongo_url = "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"
    if not mongo_url:
        raise Exception("MONGO_URL not set in environment")

    client = MongoClient(mongo_url)
    db = client["ECommerce"]
    product_collection = db["products"]
    user_data_collection = db["users"]

    products = list(product_collection.find())
    users = list(user_data_collection.find())

    product_data = [
        {
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
        }
        for p in products if p.get("isActive")
    ]

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
                "orderID": order
            })

    df_products = pd.DataFrame(product_data)
    df_user = pd.DataFrame(user_data)
    df_orders = pd.DataFrame(order_data)

    return df_products, df_user, df_orders


products, users, orders = making_data()
products = products[['name', 'category', 'price', 'description', 'productID']]

# ----------------- Vector Store -----------------
combined_text = products.to_string() + "\n" + users.to_string()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([combined_text])

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")

# ----------------- LLM -----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# ----------------- Cached Search -----------------
@lru_cache(maxsize=100)
def cached_search(query: str):
    return vector_store.similarity_search(query, k=5)

# ----------------- AI Chat -----------------
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

            Product Data : {products}
            Order Data : {orders}
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
        return {"message": result["text"]}
    except Exception as e:
        return {"message": f"Internal error: {str(e)}"}

# ----------------- Endpoints -----------------
@app.get("/chat")
async def chat(user_id: str, option: str):
    if option == "main":
        return JSONResponse({
            "message": f"Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
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
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders.tail(1).to_dict(orient='records')

    elif option == "All Orders":
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders.tail(5).to_dict(orient='records')

    elif option == "Track Order":
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders.tail(1).to_dict(orient='records')

    elif option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).",
                             "options": ["Back"]})

    elif option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:",
                             "options": ["Back"]})

    elif option == "Back":
        return await chat(user_id, "main")

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})


@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)
