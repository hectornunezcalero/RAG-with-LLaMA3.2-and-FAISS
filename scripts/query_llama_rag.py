from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from llama_run import Llama3CLI

def buscar_contexto(pregunta, db_path="vector_db"):
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    db = FAISS.load_local(db_path, modelo)
    docs = db.similarity_search(pregunta, k=3)
    return "\n\n".join([d.page_content for d in docs])

def preguntar(pregunta):
    contexto = buscar_contexto(pregunta)
    prompt = f"""Actúa como un ingeniero en ciberseguridad experto en los tipos de cifrado. Usa este contexto para responder:

{contexto}

Pregunta: {pregunta}
Respuesta:"""

    cliente = Llama3CLI(api_key="<MASTERKEY>", server_ip="127.0.0.1")  # IP local
    cliente.set_up(max_tokens=1024)
    cliente.set_prompt(prompt)
    respuesta = cliente()
    print("Respuesta:\n", respuesta)

if __name__ == "__main__":
    preguntar("¿Qué sabes sobre el Cifrado Asimétrico?")