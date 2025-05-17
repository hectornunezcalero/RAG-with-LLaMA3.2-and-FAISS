from flask import Flask, request, jsonify
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import pickle
import faiss

app = Flask(__name__)

# Cargar el modelo Llama
MODEL_PATH = "./models/Llama-3.2-1B-Instruct.Q4_K_M.gguf"
llama = Llama(model_path=MODEL_PATH, n_ctx=4096)

# Cargar la base de datos FAISS manualmente
DB_PATH = "./vector_db"
modelo_faiss = SentenceTransformer('all-MiniLM-L6-v2')
with open(f"{DB_PATH}/index.pkl", "rb") as f:
    data = pickle.load(f)

# Verificar la estructura de los datos cargados
if isinstance(data, tuple) and len(data) == 2:
    docstore, index_to_docstore_id = data
    dimension = 384  # Dimensión del modelo de embeddings
    index = faiss.IndexFlatL2(dimension)  # Crear un índice vacío si no existe
else:
    raise ValueError("El archivo index.pkl no contiene los datos esperados.")

# Inicializar FAISS con los componentes cargados
db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=modelo_faiss.encode)

@app.route("/request", methods=["POST"])
def handle_request():
    data = request.get_json()

    # Extraer parámetros
    content = data.get("content", [])
    new_prompt = data.get("new_prompt")
    task = data.get("task", "generation")
    max_tokens = int(data.get("max_tokens", 1024))

    # Buscar contexto relevante en FAISS
    query = "\n".join(content)
    docs = db.similarity_search(query, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

    # Crear prompt completo
    prompt = f"{contexto}\n\n{new_prompt}" if new_prompt else f"{contexto}\n\n{query}"

    print(f"[INFO] Prompt generado:\n{prompt}\n")

    # Llamar al modelo
    result = llama(prompt, max_tokens=max_tokens)
    respuesta = result["choices"][0]["text"]
    print("[DEBUG] Resultado crudo del modelo:", result)

    return jsonify({
        "response": respuesta.strip(),
        "status_code": 200,
        "query": data,
        "session_id": "LOCAL"
    })

if __name__ == "__main__":
    port = sum(ord(c) for c in 'llama3.2') + 5000
    print(f"Servidor corriendo en http://127.0.0.1:{port}/request")
    app.run(host="127.0.0.1", port=port)