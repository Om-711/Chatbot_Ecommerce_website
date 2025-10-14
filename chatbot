from fastapi import FastAPI
from fastapi.responses import JSONResponse
from urllib.parse import urlparse, parse_qs
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.chains import LLMChain

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

from main import making_data_endpoint
products, users = making_data_endpoint()
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCpCk8y8l3IU08n9_u_EWajQv-pibrBdps'

def get_response(user_id, option):
    if option == "main":
        return {
            "message": f" Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
            "options": ["Order Related", "Product Related", "Others"]
        }

    elif option == "Order Related":
        return {
            "message": "Please choose an option related to your orders:",
            "options": ["Recent Order", "All Orders", "Track Order", "Back"]
        }

    elif option == "Product Related":
        return {
            "message": "Need help with products? Select an option below:",
            "options": ["Request Product", "Back"]
        }

    elif option == "Others":
        return {
            "message": "You can chat with our AI assistant for general help 💬",
            "options": ["Chat with AI Assistant", "Back"]
        }

    elif option == "Recent Order":
        user_orders = users.get(user_id, {})
        if user_orders:
            return give_detail(user_orders, option)
        else:
            return {"message": "No orders found.", "options": ["Back"]}

    elif option == "All Orders":
        user_orders = users.get(user_id, {})
        if user_orders:
            return give_detail(user_orders, option)
        else:
            return {"message": "No orders found.", "options": ["Back"]}

    elif option == "Track Order":
        user_orders = users.get(user_id, {})
        if user_orders:
            return give_detail(user_orders, option)
        else:
            return {"message": "No orders to track.", "options": ["Back"]}

    elif option == "Request Product":
        return {
            "message": "Send us the product name you want to request (not available on site).",
            "options": ["Back"]
        }

    elif option == "Chat with AI Assistant":
        return chat_ai(user_id)

    elif option == "Back":
        return get_response(user_id, "main")

    else:
        return {"message": "Invalid option. Try again.", "options": ["Back"]}

def give_detail(user_data, option):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5
    )

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
    response = chain.invoke({'user_data' : user_data, 'option' : option})

    return response



def chat_ai(user_id, question):
    user_query = users.get(user_id, {}).get("query", "")
    if not user_query:
        return {"message": "No query found for user.", "options": ["Back"]}

    products, user = making_data_endpoint()

    # splitting and then embeddings
    combined_text = products.to_string() + "\n" + user.to_string()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([combined_text])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # retrival and then passed to llm for questions
    docs = vector_store.similarity_search(question, k=5)
    context = "".join(doc.page_content for doc in docs)

    

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5)
    
    prompt = PromptTemplate(
        template="""
    You are a helpful Chatbot for an E-commerce website.
    Answer all questions using only the context. If the context is insufficient, just say you don't know.
    Give all answer in accordance with the user's query and give response in short 1-2 lines only.

    {chat_history}

    Context:
    {context}

    QUESTION: {question}
    """,
        input_variables=["context", "question", "chat_history"]  
    )


        
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

  
    response = chain.invoke({'context' : context, 'question' : question})

    return response


def retriever_chain(vector_store, question):

    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5)
    
    prompt = PromptTemplate(
        template="""
    You are a helpful assistant.
    Answer all questions using only the context. If the context is insufficient, just say you don't know.
    Give all answer in English even if the video caption is in the hindi or any other langugae.

    {chat_history}

    Context:
    {context}

    QUESTION: {question}
    """,
        input_variables=["context", "question", "chat_history"]  
    )


    docs = vector_store.similarity_search(question, k=2)

    context = format_docs(docs)
        
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    response = chain.invoke({'context' : context, 'question' : question})

    return response

@app.get("/chat")
def chat(user_id: str, option: str):
    response = get_response(user_id, option)
    return JSONResponse(response)
