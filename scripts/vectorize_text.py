from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os

def vectorizar_corpus(dir_textos="./data/textos", output="vector_db"):
    modelo = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    docs = []

    for archivo in os.listdir(dir_textos):
        if archivo.endswith(".txt"):
            with open(os.path.join(dir_textos, archivo), "r", encoding="utf-8") as f:
                contenido = f.read()
                docs.append(Document(page_content=contenido))

    faiss_db = FAISS.from_documents(docs, modelo)
    faiss_db.save_local(output)
    print("[✓] Base vectorial guardada en:", output)

if __name__ == "__main__":
    vectorizar_corpus()
