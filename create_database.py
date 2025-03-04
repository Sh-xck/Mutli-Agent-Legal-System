import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database(pdf_path, db_name):
    """
    Create a FAISS database from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        db_name (str): Name for the database
    """
    print(f"\nProcessing {pdf_path}...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    os.makedirs("db", exist_ok=True)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    reader = PdfReader(pdf_path)
    print(f"Successfully opened PDF with {len(reader.pages)} pages")
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"Page {i+1}: {len(text)} characters extracted")
        if text:
            raw_text += text
            
    print(f"Total text extracted: {len(raw_text)} characters")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
    )
    texts = text_splitter.split_text(raw_text)
    print(f"Split into {len(texts)} chunks")
    print("Creating FAISS index...")
    db = FAISS.from_texts(texts, embeddings_model)
    db_path = f"db/faiss_index_{db_name}"
    print(f"Saving database to {db_path}...")
    db.save_local(db_path)
    print("Database saved successfully!")

def main():
    # Process Constitution
    create_database(
        pdf_path="tools/data/Litigation.pdf",
        db_name="litigation"
    )
    
    # Process ICAI
    create_database(
        pdf_path="tools/data/ICAI.pdf",
        db_name="ICAI"
    )

if __name__ == "__main__":
    main() 
