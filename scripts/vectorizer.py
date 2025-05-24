from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

# divide el texto en chunks para facilitar la búsqueda
def chunker(text, chunk_len, overlap):
    palabras = text.split()
    chunks = []
    for i in range(0, len(palabras), chunk_len - overlap):  # los chunks abarcan desde la primera hasta la última palabra, agrupando los chunks con solapamiento
        chunk = " ".join(palabras[i:i + chunk_len])  # une cada palabra en su/sus chunks correspondiente/es
        chunks.append(chunk)
    return chunks

# vectorización de los textos extraídos
def vectorizer(dir_textos="./txtdata", output="./vector_db", chunk_len=70sa, overlap=25):
    modelo = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # modelo de tokenización y vectorización sacado de HuggingFace
    docs = []

    # convertir los textos en documentos para su posterior chunkeo y vectorización
    for archivo in os.listdir(dir_textos):
        if archivo.endswith(".txt"):
            with open(os.path.join(dir_textos, archivo), "r", encoding="utf-8") as f:
                contenido = f.read()

                chunks = chunker(contenido, chunk_len, overlap)  # divide el texto en chunks
                print(f"Desde {archivo}, {len(chunks)} chunks generados.")
                for chunk in chunks:  # iterar sobre los chunks generados
                    docs.append(Document(page_content=chunk))  # convierte cada chunk en un objeto Document para su posterior tokenización y vectorización en FAISS

    if not docs:
        print("No se encontraron documentos válidos para vectorizar. Problema entre chunks y modelo de vectorización.")
        return

    faiss_db = FAISS.from_documents(docs, modelo)  # tokeniza y vectoriza los documentos y crea la base de datos FAISS
    print(f"Base FAISS creada con {faiss_db.index.ntotal} vectores.")
    faiss_db.save_local(output)  # guarda la base de datos FAISS en el directorio ./vector_db
    print("Base de datos vectorial actualizada en:", output)

if __name__ == "__main__":
    vectorizer()