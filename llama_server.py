from flask import Flask, request, jsonify
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import pickle
app = Flask(__name__)

MODEL_PATH = "./models/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
llama = Llama(model_path=MODEL_PATH, n_ctx=4096)

DB_PATH = "./vector_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
with open(f"{DB_PATH}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

faiss_index = faiss.read_index(f"{DB_PATH}/index.faiss")

db = FAISS(index=faiss_index,
           docstore=docstore,
           index_to_docstore_id=index_to_docstore_id,
           embedding_function=embedding_model)

@app.route("/request", methods=["POST"])
def handle_request():
    data = request.get_json()
    query = "\n".join(data.get("content", []))
    max_tokens = int(data.get("max_tokens", 1024))

    print(f"\nPregunta recibida: {query}")
    docs = db.similarity_search(query, k=1)
    print(f"Se encontró {len(docs)} chunk(s) relevante(s) para la consulta: '{query}'")
    for i, doc in enumerate(docs, start=1):
        inicio_chunk = " ".join(doc.page_content.split()[:30])  # Obtener las primeras 30 palabras de los tres chunks
        print(f"Chunk {i}: {inicio_chunk}...")

    print(f"\nProcesando respuesta...\n")
    contexto = "\n\n".join([d.page_content for d in docs])
    prompt = (f"Eres un experto sobre la información de tus documentos. "
              f"Usa el siguiente contexto para responder la pregunta que te hago después: {contexto}\n\n"
              f"La prgunta es: {query}")

    result = llama(prompt, max_tokens=max_tokens)
    respuesta = result["choices"][0]["text"].strip()

    return jsonify({
        "response": respuesta,
        "status_code": "200 OK",
        "query": data,
        "session_id": "LOCAL"
    })

if __name__ == "__main__":
    port = sum(ord(c) for c in 'llama3.2') + 5000
    app.run(host="127.0.0.1", port=port)