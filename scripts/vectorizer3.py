from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer
import os

# Carga el tokenizador del mismo modelo que usarás para embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

# divide el texto en chunks para facilitar la búsqueda
def chunker(text, chunk_len, overlap):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    for i in range(0, len(input_ids), chunk_len - overlap):  # los chunks abarcan desde la primera hasta la última palabra, agrupando los chunks con solapamiento
        chunk_ids = input_ids[i:i + chunk_len]
        chunk_offsets = offsets[i:i + chunk_len]

        # une cada palabra en su/sus chunks correspondiente/es
        start = chunk_offsets[0][0]
        end = chunk_offsets[-1][1]
        chunk_text = text[start:end].strip()
        chunks.append(chunk_text)
    return chunks

# vectorización de los textos extraídos
def vectorizer(dir_textos="./txtdata", output="./vector_db", chunk_len=150, overlap=30):
    modelo = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')  # modelo de tokenización y vectorización multilingüe sacado de HuggingFace
    docs = []

    # convertir los textos en documentos para su posterior chunkeo y vectorización
    for archivo in os.listdir(dir_textos):
        if archivo.endswith(".txt"):
            with open(os.path.join(dir_textos, archivo), "r", encoding="utf-8") as f:
                contenido = f.read()

                chunks = chunker(contenido, chunk_len, overlap)  # divide el texto en chunks
                print(f"Desde {archivo}, {len(chunks)} chunks generados.")
                for i, chunk in enumerate(chunks):
                    docs.append(Document( # convierte cada chunk en un objeto Document para su posterior tokenización y vectorización en FAISS
                        page_content=chunk,
                        metadata={"source": archivo, "chunk_index": i}
                    ))

    if not docs:
        print("No se encontraron documentos válidos para vectorizar. Problema entre chunks y modelo de vectorización.")
        return

    faiss_db = FAISS.from_documents(docs, modelo)  # tokeniza y vectoriza los documentos y crea la base de datos FAISS
    print(f"Base FAISS creada con {faiss_db.index.ntotal} vectores.")
    faiss_db.save_local(output)  # guarda la base de datos FAISS en el directorio ./vector_db
    print("Base de datos vectorial actualizada en:", output)

if __name__ == "__main__":
    vectorizer()