# MVP Learning App

A learning application that uses LangChain and OpenAI embeddings to create a semantic search over markdown content.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd mvp-learning-app
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install langchain langchain-community python-dotenv faiss-cpu tiktoken
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
mvp-learning-app/
├── content/           # Directory containing markdown files
├── build_embeddings.py    # Script to build vector store from content
├── .env              # Environment variables (not in git)
├── .gitignore       # Git ignore file
└── vectorstore.pkl   # Generated vector store (not in git)
```

## Usage

1. Add your markdown content files to the `content/` directory.

2. Build the vector store:
```bash
python build_embeddings.py
```
This script will:
- Load markdown documents from the `content` directory
- Split them into smaller chunks
- Create embeddings using OpenAI's embedding model
- Build a FAISS vector store
- Save the vector store to disk as `vectorstore.pkl`

## Development

- The project uses `langchain` for document loading and processing
- OpenAI's `text-embedding-ada-002` model is used for creating embeddings
- FAISS is used for efficient similarity search

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Notes

- The vector store file (`vectorstore.pkl`) is not tracked in git due to its size
- Make sure to keep your `.env` file secure and never commit it to version control
