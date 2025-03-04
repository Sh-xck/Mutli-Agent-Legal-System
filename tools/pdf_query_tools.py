from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import tool
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


@tool
def Litigation_pdf_query(query: str) -> str:
    """
    Searches the FAISS vector database for relevant legal information
    related to the given query and returns the top-ranked results.
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    db = FAISS.load_local("db/faiss_index_litigation",
                          embeddings_model, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)
    return result


@tool
def ICAI_pdf_query(query: str) -> str:
    """
    Queries the FAISS database for ICAI-related legal information
    and returns the most relevant sections.
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    db = FAISS.load_local("db/faiss_index_ICAI",
                          embeddings_model, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)
    return result
