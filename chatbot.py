from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
import os
from main import making_data_endpoint
from langchain.embeddings import SentenceTransformerEmbeddings

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

products, users = making_data_endpoint()
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCpCk8y8l3IU08n9_u_EWajQv-pibrBdps'  

async def give_detail_async(user_data, option):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        You are given user data in dataframe format. you have to give ans about this options in short 1-2 lines only.

        user data : {user_data}
        option : {option}
        """,
        input_variables=["user_data", "option"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = await chain.ainvoke({"user_data": user_data, "option": option})
    
    return {"message": result["text"] if isinstance(result, dict) and "text" in result else str(result)}

async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    try:
        products, user = making_data_endpoint()

        combined_text = products.to_string() + "\n" + user.to_string()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([combined_text])

        # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)

        docs = vector_store.similarity_search(question, k=5)
        context = "\n".join([d.page_content for d in docs])

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

        prompt = PromptTemplate(
            template="""
            You are a helpful Chatbot for an E-commerce website.
            Answer all questions using only the provided context.
            If the context is insufficient, reply with "No data found".
            Respond concisely in 1–2 lines only.

            Context:
            {context}

            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.ainvoke({"context": context, "question": question})

        return {"message": result["text"] if isinstance(result, dict) and "text" in result else str(result)}

    except Exception as e:
       
        print("Error in chat_ai_async:", repr(e))
        return {"message": f"Internal error: {str(e)}"}



@app.get("/chat")
async def chat(user_id: str, option: str):
    
    if option == "main":
        return JSONResponse({
            "message": f" Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
            "options": ["Order Related", "Product Related", "Others"]
        })
    if option == "Order Related":
        return JSONResponse({"message": "Please choose an option related to your orders:",
                             "options": ["Recent Order", "All Orders", "Track Order", "Back"]})
    if option == "Product Related":
        return JSONResponse({"message": "Need help with products? Select an option below:",
                             "options": ["Request Product", "Back"]})
    if option == "Others":
        return JSONResponse({"message": "You can chat with our AI assistant for general help 💬",
                             "options": ["Chat with AI Assistant", "Back"]})

    if option in ("Recent Order", "All Orders", "Track Order"):
        user_orders = users.get(user_id, {})
        if user_orders:
            resp = await give_detail_async(user_orders, option)
            return JSONResponse(resp)
        else:
            return JSONResponse({"message": "No orders found.", "options": ["Back"]})

    if option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).",
                             "options": ["Back"]})

    if option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:",
                             "options": ["Back"]})

    if option == "Back":
        return JSONResponse(await chat(user_id, "main"))

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})



@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)
