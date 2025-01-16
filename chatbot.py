from flask import Flask, request, jsonify
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Explicitly set API key in the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

# Load and preprocess the PDF
loader = PyPDFLoader("HPV.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Initialize FAISS vectorstore
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001"),
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize language model
llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

# Define RAG logic
def get_rag_response(query):
    system_prompt = (
        "You are a question-answering assistant. Use the retrieved context to answer the user's question. "
        "If unsure, say you don't know. Keep your response concise, using up to three sentences.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# Root route for testing
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "RAG model is running! Use the '/ask' endpoint to send queries."})

# API Endpoint for RAG
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query not provided"}), 400
    try:
        answer = get_rag_response(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
