import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from uuid import uuid4
from langchain_core.documents import Document
from pathlib import Path

def customDocLoad():
    """Load or create the FAISS database."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    faiss_index_path = "faiss_index"

    if os.path.exists(faiss_index_path):
        print("FAISS index exists. Loading it...")
        vector_store = FAISS.load_local(faiss_index_path, embeddings,allow_dangerous_deserialization=True)
        documents = []  # Documents are not needed for querying after loading the index
    else:
        print("FAISS index does not exist. Creating it...")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Load documents from CSV
        data = pd.read_csv("data.csv", encoding="ISO-8859-1")
        documents = []
        for _, entry in data.iterrows():
            content = str(entry[1])
            mtd = {"source": entry[0], "id": str(uuid4())}  # Add UUID directly to metadata
            document = Document(page_content=content, metadata=mtd)
            documents.append(document)

        # Add documents to the vector store
        vector_store.add_documents(documents)
        vector_store.save_local(faiss_index_path)

    return vector_store


def query(vector_store):
    """Query the FAISS index and write results to a file."""
    filepath = "results/results.txt"

    Path("results").mkdir(parents=True, exist_ok=True)
    try:
        os.remove(filepath)
    except OSError:
        pass

    

    with open(filepath, "w", encoding="ISO-8859-1") as file:
        # Iterate through all documents in the vector store
        for doc_id, document in vector_store.docstore._dict.items():
            current_project = document.metadata.get("source", None)
            file.write(f"\nCURRENT DOC: {document.page_content} {document.metadata}\n")

            # Perform similarity search
            results = vector_store.similarity_search_with_score(
                document.page_content,
                k=3,
            )

            # Filter results based on project metadata
            filtered_results = [
                res for res in results if res[0].metadata.get("source") != current_project
            ]
            if filtered_results:
                for res in filtered_results:
                    result_text = str(res)
                    file.write(result_text + "\n")
   


# Main execution
vector_store = customDocLoad()
query(vector_store)