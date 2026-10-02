import os

from dotenv import load_dotenv
from pinecone import Pinecone

from huggingface_hub import InferenceClient

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# Hugging Face Remote Embeddings
# ============================================================

class HuggingFaceRemoteEmbeddings(Embeddings):

    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN
        )

        self.model = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_query(self, text: str) -> list[float]:

        result = self.client.feature_extraction(
            text,
            model=self.model
        )

        # Convert numpy array → Python list
        result = result.tolist()

        # Single sentence normally returns [[384 values]]
        if isinstance(result[0], list):
            result = result[0]

        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        result = self.client.feature_extraction(
            texts,
            model=self.model
        )

        return result.tolist()


# Create remote embedding object
embedding = HuggingFaceRemoteEmbeddings()


# ============================================================
# Pinecone
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "medical-chatbot"


# Connect to EXISTING Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)


# ============================================================
# Retriever
# ============================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# Gemini
# ============================================================

chatModel = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


# ============================================================
# Format documents
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# Get relevant documents
# ============================================================

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

    # This sends the search query to:
    #
    # Hugging Face
    #       ↓
    # MiniLM embedding
    #       ↓
    # 384-dimensional vector
    #       ↓
    # Pinecone
    #
    docs = retriever.invoke(search_query)

    return format_docs(docs)


# ============================================================
# Prompt
# ============================================================

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


# ============================================================
# RAG Chain
# ============================================================

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