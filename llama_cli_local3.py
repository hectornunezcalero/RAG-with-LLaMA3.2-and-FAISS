import requests
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
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
                  f"La pregunta del usuario a contestar es:\n{question}\n\n")

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
        self.window.geometry("900x700")
        self.window.configure(bg="#f0f2f5")

        self.build_interface()

    def build_interface(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=('Segoe UI', 10), padding=6)
        style.configure("TLabel", font=('Segoe UI', 10), background="#f0f2f5")
        style.configure("TFrame", background="#f0f2f5")

        # Título
        title = ttk.Label(self.window, text="🧠 Llama3.2 - RAG Assistant", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(20, 10))

        # Frame principal
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Pregunta
        label_input = ttk.Label(main_frame, text="🔍 Escribe tu pregunta:")
        label_input.pack(anchor="w")

        self.input_text = scrolledtext.ScrolledText(main_frame, height=6, font=('Segoe UI', 10), wrap=tk.WORD)
        self.input_text.pack(fill="x", pady=(5, 15))

        # Botón de enviar - con estilo mejorado
        self.send_button = tk.Button(
            main_frame,
            text="🚀 Enviar consulta",
            font=('Segoe UI', 11, 'bold'),
            bg="#4a90e2",
            fg="white",
            activebackground="#357ABD",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
            command=self.send_request
        )
        self.send_button.pack(pady=(0, 10))

        # Separador
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=10)

        # Respuesta
        label_output = ttk.Label(main_frame, text="📬 Respuesta generada:")
        label_output.pack(anchor="w")

        self.output_text = scrolledtext.ScrolledText(main_frame, height=15, font=('Segoe UI', 10), wrap=tk.WORD, bg="#ffffff")
        self.output_text.pack(fill="both", expand=True)

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