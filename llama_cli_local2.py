import requests
import logging
import tkinter as tk
from tkinter import scrolledtext
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
import tiktoken  # Para contar tokens

# Configuración
LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SERVER_IP = "127.0.0.1"
API_KEY = "abc123"
VECTOR_DB_PATH = "./vector_db"
MAX_TOKENS = 4096
RESERVED_FOR_QUESTION = 512

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    return len(tokenizer.encode(text))

# Cargar base vectorial
with open(f"{VECTOR_DB_PATH}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

faiss_index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

db = FAISS(index=faiss_index, docstore=docstore,
           index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model)


class Llama3CLI:
    def __init__(self):
        self.session_id = "0"

    def process_request(self, question: str):
        print(f"Resolviendo pregunta: {question}")

        # Recuperamos chunks por similitud
        docs = db.similarity_search(question, k=5)
        print(f"Chunks rescatados por similitud: {len(docs)}")
        for doc in docs:
            chunk_preview = " ".join(doc.page_content.split()[:20]) + " ..."
            print(f" - {doc.metadata['source']} (chunk {doc.metadata['chunk_index']}): {chunk_preview}")

        contexto = ""
        for doc in docs:
            chunk = doc.page_content
            contexto += chunk + "\n------\n"

        prompt = (f"Eres un asistente experto sobre la información de tus documentos. "
                  f"Usa el siguiente contexto y contesta a la pregunta:\n\n"
                  f"Contexto:\n{contexto}\n\n"
                  f"La pregunta del usuario a contestar es:\n{question}\n\n"
                  f"Responde a ella claro y concisa")

        data = {
            "content": [question],
            "new_prompt": prompt,
            "task": "generation",
            "pooling": "none",
            "max_tokens": MAX_TOKENS
        }

        headers = {"Authorization": API_KEY, "Session": self.session_id}
        url = f"http://{SERVER_IP}:{LLAMA_PORT}/request"
        print("Enviando prompt al LLM del servidor...")

        try:
            response = requests.post(url, json=data, headers=headers)
        except Exception as ex:
            logging.error(f"Connection error: {ex}")
            return {"response": "No se pudo conectar al servidor", "status_code": "Host Unreachable"}

        if response.status_code == 200:
            resp_json = response.json()
            self.session_id = resp_json.get("session_id", "0")
            return resp_json
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            return {"response": "Error del servidor", "status_code": response.status_code}


class Llama3GUI:
    def __init__(self):
        self.client = Llama3CLI()
        self.window = tk.Tk()
        self.window.title("Llama3.2 - RAG Assistant")

        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(2, weight=1)

        self.input_text = scrolledtext.ScrolledText(self.window, height=10, width=70)
        self.input_text.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

        self.send_button = tk.Button(self.window, text="Send Request", command=self.send_request)
        self.send_button.grid(row=1, column=0, columnspan=2, pady=5, padx=10)

        self.output_text = scrolledtext.ScrolledText(self.window, height=20, width=70)
        self.output_text.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

    def send_request(self):
        pregunta = self.input_text.get('1.0', tk.END).strip()
        if not pregunta:
            self.output_text.insert(tk.END, "Escribe una pregunta.\n")
            return

        respuesta = self.client.process_request(pregunta)
        print("Consulta respondida sobre:", pregunta)
        print("--------------------------")
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, respuesta.get("response", "Sin respuesta."))

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = Llama3GUI()
    app.run()

