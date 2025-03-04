import os
import pickle

import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from streamlit.components.v1 import html


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

    st.title("Welcome to Marczel AI’s Hypnosis Hub")

    # Explanation or instructions
    st.write(
        "Discover guided hypnosis sessions for wealth, health, and relationships. Ask your question to begin your transformative journey."
        "Empower your mind and unlock your full potential with Marczell AI’s guided hypnosis. Ask anything and discover the path to a richer life."
    )

    # User input
    user_query = st.text_input("Type your hypnosis question or goal here...")
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

    # Add Eleven Labs widget
    eleven_labs_html = """
    <elevenlabs-convai agent-id="kb5KeSOK5l9WpYhuK0Iz"></elevenlabs-convai>
    <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
    """
    html(eleven_labs_html, height=175)


if __name__ == "__main__":
    main()
