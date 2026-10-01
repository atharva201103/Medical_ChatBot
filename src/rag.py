import os

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)


# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Gemini
chatModel = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


# Format documents
def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# Get relevant documents
def get_context(data):

    question = data["input"]
    history = data["history"]

    if history:
        search_query = f"""
Previous conversation:
{history}

Current question:
{question}
"""
    else:
        search_query = question

    docs = retriever.invoke(search_query)

    return format_docs(docs)


# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

Answer the user's question using the provided medical
documents and conversation history.

Conversation history:
{history}

Medical documents:
{context}

Current question:
{input}

Important instructions:

1. Use the medical documents as the primary source.
2. Use the conversation history to understand references
   such as "it", "this", "that", or "the condition".
3. If the answer is not available in the provided medical
   documents, say:

"I don't have enough information in the provided medical documents."

4. Do not invent medical information.
5. Give clear and simple explanations.
""")


# RAG chain
rag_chain = (
    {
        "context": RunnableLambda(get_context),
        "history": lambda x: x["history"],
        "input": lambda x: x["input"]
    }
    | prompt
    | chatModel
    | StrOutputParser()
)