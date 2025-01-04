import os
import pickle

import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS


def load_vector_store():
    # Read the saved FAISS index
    index = faiss.read_index("embeddings/faiss_index.index")

    # Load the VectorStore object from pickle
    with open("embeddings/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    # Attach the index to the stored object
    store.index = index
    return store


def main():
    # Load environment variables
    load_dotenv()

    st.title("Local Learning Q&A")

    # Explanation or instructions
    st.write(
        "Ask a question about the content in `content/` folder and get an AI-powered answer."
    )

    # User input
    user_query = st.text_input("Enter your question here...")

    # On button click, process the query
    if st.button("Get Answer"):
        if user_query.strip():
            # Load the vector store
            vector_store = load_vector_store()

            # Create an LLM object (OpenAI with default GPT-3.5 settings)
            llm = OpenAI(temperature=0)

            # Create a retriever
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )

            # Create a RetrievalQA chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # "stuff" = combine retrieved docs into a single prompt
                retriever=retriever,
            )

            # Run the chain to get an answer
            answer = chain.run(user_query)
            st.write("**Answer:**", answer)
        else:
            st.warning("Please enter a question before clicking 'Get Answer'.")


if __name__ == "__main__":
    main()
