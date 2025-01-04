import os
import pickle

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import faiss


def build_vector_store():
    # Load environment variables (OPENAI_API_KEY)
    load_dotenv()

    # 1. Load documents from `content` folder
    loader = DirectoryLoader(
        'content',
        glob='**/*.md',
        loader_cls=TextLoader
    )  
    docs = loader.load()

    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # 3. Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()  # uses text-embedding-ada-002 by default

    # 4. Build a local FAISS vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 5. Save the index files locally
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    # Write the faiss index to a file
    faiss.write_index(vectorstore.index, "embeddings/faiss_index.index")

    # Save the VectorStore object (minus the index) as a pickle
    vectorstore.index = None  # to avoid pickling the index directly
    with open("embeddings/faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    build_vector_store()
