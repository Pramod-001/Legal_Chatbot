import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

START_URL = "https://www.indiacode.nic.in/handle/123456789/1362" 
DB_FAISS_PATH = "vectorstore/db_faiss"

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
  
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    text = soup.get_text()
    return re.sub(r"\n\n+", "\n\n", text).strip()

def ingest_from_web_recursive():
    print(f" Starting recursive crawl at: {START_URL}")

    loader = RecursiveUrlLoader(
        url=START_URL, 
        max_depth=3, 
        extractor=bs4_extractor,
        prevent_outside=True
    )

    print(" Crawling and loading pages... (This might take a few minutes)")
    documents = loader.load()
    print(f" Successfully loaded {len(documents)} web pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f" Split into {len(texts)} chunks.")

    print(" Creating Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(texts, embeddings)
    
    db.save_local(DB_FAISS_PATH)
    print(f" Success! Legal database built from web crawl.")

if __name__ == "__main__":
    ingest_from_web_recursive()