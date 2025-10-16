# chatbot.py
import os
import re
import asyncio
from functools import lru_cache
from functools import partial
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# prefer community embeddings import (avoid deprecation warning)
try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except Exception:
    # fallback if the package layout differs
    from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import pandas as pd
from pymongo import MongoClient

# ============ Configuration / Globals ============
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://apnabzaar.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# put your key in env — you already had this
os.environ["GOOGLE_API_KEY"] = os.environ.get(
    "GOOGLE_API_KEY", "AIzaSyAIzbhiQ1Ga-XfzozyoYugrrhwAXtdrxB8"
)

# Globals that will be populated by heavy_init()
products = None  # pandas DataFrame
users = None
orders = None
vector_store = None
embeddings = None

# LLM can be initialized at import time (lightweight)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# ============ Data loader (your existing logic) ============
def making_data():
    mongo_url = (
        "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"
    )
    client = MongoClient(mongo_url)
    db = client["ECommerce"]
    product_collection = db["products"]
    user_data_collection = db["users"]

    products = list(product_collection.find())
    users = list(user_data_collection.find())

    product_data = []
    for p in products:
        if p.get("isActive"):
            product_data.append(
                {
                    "productID": str(p["_id"]),
                    "name": p.get("name", ""),
                    "price": p.get("price", ""),
                    "category": p.get("category", ""),
                    "description": p.get("description", ""),
                    "images": p.get("images", "Not Found"),
                    "stock": p.get("stock", "0"),
                    "rating": p.get("rating", "0"),
                    "reviews": p.get("reviews", "0"),
                    "createdAt": p.get("createdAt", ""),
                    "updatedAt": p.get("updatedAt", ""),
                    "isActive": p.get("isActive", True),
                }
            )

    user_data = []
    order_data = []
    for u in users:
        for history in u.get("history", []):
            user_data.append(
                {
                    "user_id": str(u["_id"]),
                    "productID": str(history.get("productId", "")),
                    "event": history.get("event", {}).get("type", "Not Found"),
                    "Timestamp": history.get("time", ""),
                    "duration": history.get("duration", 0) / 1000,  # ms -> s
                }
            )
        for order in u.get("orders", []):
            order_data.append({"user_id": str(u["_id"]), "orderID": (order)})

    df_products = pd.DataFrame(product_data)
    df_user = pd.DataFrame(user_data)
    df_orders = pd.DataFrame(order_data)

    return df_products, df_user, df_orders


# ============ Heavy initialization (run in background) ============
def heavy_init():
    """
    Runs synchronously in a thread/executor. Populates global products/users/orders,
    embeddings and vector_store. This prevents blocking import/startup.
    """
    global products, users, orders, embeddings, vector_store

    try:
        print("[heavy_init] Starting heavy initialization (DB read, FAISS, embeddings)...")
        products_local, users_local, orders_local = making_data()

        # ensure productID present and consistent with your prompt requirements
        if "productID" not in products_local.columns:
            products_local["productID"] = products_local.index.astype(str)

        # keep the exact columns you want
        products_local = products_local[
            ["productID", "name", "category", "price", "description"]
        ].copy()

        combined_text = products_local.to_string() + "\n" + users_local.to_string()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([combined_text])

        # instantiate embeddings (may be heavy)
        embeddings_local = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # load or build FAISS index
        try:
            vs = FAISS.load_local("faiss_index", embeddings_local)
            print("[heavy_init] Loaded existing FAISS index from faiss_index/")
        except Exception as e:
            print("[heavy_init] Could not load local FAISS index, building new one...", e)
            vs = FAISS.from_documents(chunks, embeddings_local)
            vs.save_local("faiss_index")
            print("[heavy_init] Saved new FAISS index to faiss_index/")

        # assign to globals
        products, users, orders = products_local, users_local, orders_local
        embeddings, vector_store = embeddings_local, vs

        # clear any cached_search cache (if used)
        try:
            _cached_search.cache_clear()
        except Exception:
            pass

        print("[heavy_init] Initialization finished — service ready.")
    except Exception as e:
        print("[heavy_init] ERROR during heavy_init:", repr(e))


@app.on_event("startup")
async def startup_event():
    """
    Schedule heavy_init in the default executor so the server can bind its port immediately.
    """
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, heavy_init)
    print("[startup] Server started and accepting connections; heavy init scheduled.")


# ============ Readiness helper ============
def is_ready():
    return vector_store is not None and products is not None and orders is not None


# ============ Simple cached search (LRU) ============
# cache searches by query text; results are the list returned by FAISS
@lru_cache(maxsize=256)
def _cached_search(query: str):
    # assumes vector_store is ready before calling
    return vector_store.similarity_search(query, k=5)


def cached_search(query: str):
    if not is_ready():
        raise RuntimeError("Service not ready")
    return _cached_search(query)


# ============ Product-finding helpers ============
def find_product_by_name_in_question(question: str, products_df: pd.DataFrame):
    """
    Simple case-insensitive substring/token-based match for product name.
    Returns the first match dict with productID, name, price or None.
    """
    if products_df is None or products_df.empty:
        return None
    q = question.lower().strip()
    # direct substring match
    mask = products_df["name"].str.lower().apply(lambda n: q in n or n in q)
    matches = products_df[mask]
    if matches.empty:
        # token match fallback
        tokens = re.findall(r"\w+", q)
        for t in tokens:
            if len(t) < 3:
                continue
            m = products_df[products_df["name"].str.lower().str.contains(re.escape(t))]
            if not m.empty:
                matches = m
                break
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {"productID": str(row["productID"]), "name": row["name"], "price": row["price"]}


def format_price_response(product: dict):
    # required exact format: price, Name of product and the product link in this exact format:
    # https://apnabzaar.netlify.app/productdetail/product_id
    link = f"https://apnabzaar.netlify.app/productdetail/{product['productID']}"
    return f"{product['price']}, {product['name']}, {link}"


# ============ Chat logic (uses vector search + LLM) ============
async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    # if still warming up:
    if not is_ready():
        return {"message": "Service warming up — please try again in a few seconds.", "options": ["Back"]}

    try:
        # short-circuit price requests and return exact required format programmatically
        price_keywords = ["price", "cost", "how much", "what is the price", "price of"]
        if any(k in question.lower() for k in price_keywords):
            product = find_product_by_name_in_question(question, products)
            if product:
                return {"message": format_price_response(product)}
            # if we couldn't find product but user asked price — return apology per your rules
            return {
                "message": (
                    "We are sorry, the product you requested is currently not available on our site. "
                    "However, we value your interest and would be happy to assist you with similar products or alternatives "
                    "that meet your needs. Please let us know what you're looking for, and we'll do our best to help you find a suitable option."
                )
            }

        # similarity search (cached)
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
            "

            Context:
            {context}

            Question: {question}

            Product Data : {products}
            Order Data : {orders}
            """,
            input_variables=["context", "question", "products", "orders"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        # pass product/order tables as strings like before
        result = await chain.ainvoke(
            {
                "context": context,
                "question": question,
                "products": products.to_string(),
                "orders": orders.to_string(),
            }
        )

        # handle LLMChain return shape
        if isinstance(result, dict) and "text" in result:
            return {"message": result["text"]}
        return {"message": str(result)}
    except Exception as e:
        print("Error in chat_ai_async:", repr(e))
        return {"message": f"Internal error: {str(e)}"}


# ============ Routes (preserve your behavior, but return JSON-serializable responses) ============
@app.get("/chat")
async def chat(user_id: str, option: str):
    # ensure readiness for endpoints that rely on orders/products
    if option == "main":
        return JSONResponse(
            {
                "message": f" Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
                "options": ["Order Related", "Product Related", "Others"],
            }
        )
    elif option == "Order Related":
        return JSONResponse(
            {
                "message": "Please choose an option related to your orders:",
                "options": ["Recent Order", "All Orders", "Track Order", "Back"],
            }
        )
    elif option == "Product Related":
        return JSONResponse(
            {"message": "Need help with products? Select an option below:", "options": ["Request Product", "Back"]}
        )
    elif option == "Others":
        return JSONResponse(
            {"message": "You can chat with our AI assistant for general help 💬", "options": ["Chat with AI Assistant", "Back"]}
        )

    # The endpoints that return order data should handle warming-up state
    if option in ("Recent Order", "All Orders", "Track Order"):
        if not is_ready():
            return JSONResponse({"message": "Service warming up — please try again in a few seconds.", "options": ["Back"]})

        user_id_orders = orders[orders["user_id"] == user_id] if not orders.empty else pd.DataFrame()
        if user_id_orders.empty:
            return JSONResponse({"message": "No data found", "options": ["Back"]})

        if option == "Recent Order" or option == "Track Order":
            subset = user_id_orders.tail(1)
        else:  # All Orders
            subset = user_id_orders.tail(5)

        # return list of records for JSON serialization (keeps behavior similar)
        return JSONResponse(subset.to_dict(orient="records"))

    elif option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).", "options": ["Back"]})

    elif option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:", "options": ["Back"]})

    elif option == "Back":
        # call main menu
        return JSONResponse(await chat(user_id, "main"))

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})


@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)


# allow running locally with proper port binding
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chatbot:app", host="0.0.0.0", port=port, log_level="info")
