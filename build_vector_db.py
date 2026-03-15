
import os
import shutil
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re

#  Config 
VECTOR_DB_PATH = "faiss_index"   # single consistent path
PDF_PATH       = "Machine-Learning-Systems.pdf"
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 100
BATCH_SIZE     = 64

# TDS article URLs to scrape 
TDS_URLS = [
    # Machine Learning
    "https://towardsdatascience.com/machine-learning-introduction-a-comprehensive-guide-af6712cf68a3/",
    "https://towardsdatascience.com/how-id-learn-machine-learning-if-i-could-start-over-c68d697e6a8a/",
    "https://towardsdatascience.com/5-beginner-friendly-steps-to-learn-machine-learning-and-data-science-with-python-bf69e211ade5",
    "https://towardsdatascience.com/machine-learning-in-production-what-this-really-means/",
    "https://towardsdatascience.com/2025-must-reads-agents-python-llms-and-more/",
    # Deep Learning
    "https://towardsdatascience.com/neural-networks-a-beginners-guide-7b374b66441a/",
    "https://towardsdatascience.com/neural-networks-illustrated-part-1-how-does-a-neural-network-work-c3f92ce3b462/",
    "https://towardsdatascience.com/deep-learning-illustrated-part-2-how-does-a-neural-network-learn-481f70c1b474/",
    "https://towardsdatascience.com/the-basics-of-deep-learning-with-pytorch-in-1-hour/",
    "https://towardsdatascience.com/understanding-convolutional-neural-networks-cnns-through-excel/",
    # NLP
    "https://towardsdatascience.com/transformers-89034557de14",
    "https://towardsdatascience.com/beautifully-illustrated-nlp-models-from-rnn-to-transformer-80d69faf2109/",
    "https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/",
    "https://towardsdatascience.com/practical-introduction-to-transformer-models-bert-4715ed0deede/",
    "https://towardsdatascience.com/transform-your-nlp-game-17f5bd0d87ea/",
    # LLM & RAG
    "https://towardsdatascience.com/retrieval-augmented-generation-rag-an-introduction/",
    "https://towardsdatascience.com/retrieval-augmented-generation-intuitively-and-exhaustively-explain-6a39d6fe6fc9/",
    "https://towardsdatascience.com/a-beginners-guide-to-building-a-retrieval-augmented-generation-rag-application-from-scratch-e52921953a5d/",
    "https://towardsdatascience.com/how-to-make-your-llm-more-accurate-with-rag-fine-tuning/",
    "https://towardsdatascience.com/beyond-rag/",
]

#  Text cleaner
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    junk = [r"Sign in", r"Submit an Article", r"Write For TDS",
            r"Toggle.*", r"LinkedIn", r"\bX\b"]
    for p in junk:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

#  Delete old index 
if os.path.exists(VECTOR_DB_PATH):
    shutil.rmtree(VECTOR_DB_PATH)
    print(f"Deleted old index at {VECTOR_DB_PATH}")

# Load PDF 
print("Loading PDF...")
pdf_docs = PyPDFLoader(PDF_PATH).load()
print(f"PDF pages loaded: {len(pdf_docs)}")

#  Load and clean TDS web articles 
print("Loading TDS articles...")
try:
    web_docs = WebBaseLoader(TDS_URLS).load()
    clean_web_docs = [
        Document(page_content=clean_text(d.page_content), metadata=d.metadata)
        for d in web_docs
    ]
    print(f"Web articles loaded: {len(clean_web_docs)}")
except Exception as e:
    print(f"Web scraping failed (skipping): {e}")
    clean_web_docs = []

# Combine and split 
all_docs = pdf_docs + clean_web_docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# FAISS with HuggingFace embeddings 
print("Loading embedding model (downloading ~90MB first time)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Building FAISS index...")
vectorstore = None

for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
    batch = chunks[i : i + BATCH_SIZE]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)

#  Save index 
vectorstore.save_local(VECTOR_DB_PATH)
print(f"\nFAISS index saved to: {VECTOR_DB_PATH}/")
print("Now commit faiss_index/ to GitHub and deploy on Render.")