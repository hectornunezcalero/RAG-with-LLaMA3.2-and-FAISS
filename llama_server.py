from flask import Flask, request, jsonify
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import pickle
import faiss

app = Flask(__name__)

MODEL_PATH = "./models/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
llama = Llama(model_path=MODEL_PATH, n_ctx=4096)

DB_PATH = "./vector_db"
modelo_faiss = SentenceTransformer('all-MiniLM-L6-v2')
with open(f"{DB_PATH}/index.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, tuple) and len(data) == 2:
    docstore, index_to_docstore_id = data
    first_doc_id = next(iter(docstore._dict.keys()))
    example_vector = modelo_faiss.encode(docstore._dict[first_doc_id].page_content)
    dimension = len(example_vector)
    index = faiss.IndexFlatL2(dimension)
else:
    raise ValueError("El archivo index.pkl no es válido.")

db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=modelo_faiss.encode)

@app.route("/request", methods=["POST"])
def handle_request():
    data = request.get_json()
    query = "\n".join(data.get("content", []))
    max_tokens = int(data.get("max_tokens", 1024))

    docs = db.similarity_search(query, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = (f"Eres un experto en investigación sobre información de tus documentos. "
              f"Usa este contexto para responder la pregunta del usuario:\n\n{contexto}\n\nPregunta: {query}\nRespuesta:")
    print("\nPrompt inicial generado:\n", prompt)

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