# -*- coding: utf-8 -*-
import os
import numpy as np
import faiss
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import notebook_login

# Initialize directories
!mkdir docs

# Load documents
documents = []
for file in os.listdir("docs"):
    file_path = os.path.join("docs", file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(('.docx', '.doc')):
        loader = Docx2txtLoader(file_path)
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        continue
    documents.extend(loader.load())

# Split documents into chunks
document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks = document_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create document embeddings
document_texts = [doc.page_content for doc in document_chunks]
document_embeddings = np.array([embedding_model.encode(doc) for doc in document_texts])

# Create and save FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)
faiss.write_index(index, 'faiss_index.index')

# Save document chunks
with open('document_chunks.npy', 'wb') as f:
    np.save(f, document_texts)

# Authenticate Hugging Face Hub
notebook_login()

# Initialize the text generation model and pipeline
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def query_knowledge_base(query, top_n=3):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    results = sorted([(document_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])], key=lambda x: x[1])
    
    print("Top relevant document chunks:")
    for i, (doc, score) in enumerate(results):
        print(f"Rank {i+1}:\nDocument: {doc}\nScore: {score}\n{'-'*50}")
        
    top_documents = " ".join([doc for doc, _ in results])
    return top_documents

def query_llama(query, context):
    prompt = f"\n\nBased on the following information:\n{context}\n\nPlease provide a conversational answer to the question and be crisp and on point: {query}"
    result = pipe(prompt, max_new_tokens=200)
    return result[0]['generated_text']

if __name__ == "__main__":
    print("Enter 'exit' to stop the program.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting the program.")
            break

        context = query_knowledge_base(query, top_n=3)
        answer = query_llama(query, context)
        print("\n\033[92mAnswer:\033[0m", answer)
