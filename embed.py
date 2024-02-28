import os
import sys
import yaml
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from doc_loader import load_files
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_config(): 
    """Function to load configuration from config.yaml file.

    Returns: 
        dict: items in config.yaml
    """
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: File config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as err:
        print(f"Error reading YAML file: {err}")
        sys.exit(1)

    return data


    
def create_embeddings(model, texts):  
    """Function to create embeddings for documents.

    Args:
        model: The embeddings model.
        documents: The list of documents.

    Returns:
        list: The list of embeddings for each document.
    """
    data_to_upload = []
    for i, text in enumerate(texts):
        embeddings = model.embed_documents(text)
        data_to_upload.append((str(i), embeddings[0]))

    return data_to_upload
    


def initialize_index(index_name: str, config: dict, pinecone_api_key: str = None):
    """Function to initialize the Pinecone index.

    Args:
        index_name (str): The name of the index.
        config (dict): The configuration dictionary.

    Returns:
        Pinecone.Index: The initialized index.
    """
    pc = Pinecone(
        api_key= pinecone_api_key
    )
    index_name = config["pc"]["index_name"]
    # create index if not already created
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=config["pc"]["dimension"],
            metric=config["pc"]["metric"],
            spec=ServerlessSpec(
                cloud=config["pc"]["spec"]["cloud"],
                region=config["pc"]["spec"]["region"]
            )
        )

    # initialize the index
    index = pc.Index(index_name)
    
    # index.describe_index_stats()
    return index


if __name__ == "__main__":
    config = load_config()
    model_name = config["openai"]["embeddings_model"]
    model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    text = load_files(config["data_dir"])
    embeddings = create_embeddings(model, text)
    # print(f"Embedding {len(documents)} documents, this may take a while...")

    index = initialize_index(config["pc"]["index_name"], config)
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    index.upsert(vectors=embeddings, ids=[str(i) for i in range(len(embeddings))])

    print(index.describe_index_stats()) 
