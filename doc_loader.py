import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    texts = [str(doc) for doc in documents]
    return texts

def load_files(directory):
    texts = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r') as file:
                text = file.read()
                text = preprocess_text(text)
                texts.append(text)
        elif file_name.endswith('.pdf'):
            text = process_pdf(file_path)
            texts.append(text)
    return texts

