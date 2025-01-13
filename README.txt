# FAISS-based Document Similarity Search

This project demonstrates the creation and querying of a FAISS-based vector store for document similarity search. It uses embeddings from Hugging Face's `sentence-transformers` and stores metadata about documents for advanced querying.

## Features
- Load or create a FAISS vector index for document similarity search.
- Add documents from a CSV file to the vector store.
- Perform similarity search on the vector store and filter results based on metadata.
- Save query results to a text file.

## Prerequisites
- Python 3.8 or higher
- A CSV file named `data.csv` in the same directory as the script. This file should contain at least two columns: 
  - The first column is used as the document source (metadata).
  - The second column is the content of the document.

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt