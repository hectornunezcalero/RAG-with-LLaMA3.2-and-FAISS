# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación Aumentada por Recuperación (RAG)      #
#           con LLaMA 3.2 como asistente para consultas                 #
#           sobre artículos farmacéuticos que dispone el                #
#           grupo de investigación de la Universidad de Alcalá.         #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: llama_client.py                                         #
#       Funciones principales:                                          #
#        1. Prestar la GUI con Tkinter para interactuar con LLaMA3.2 3B #
#        2. Búscar documentos relacionados en la base de datos FAISS    #
#        3. Enviar consultas al servidor que dispone del LLM            #
#        4. Visualizar y poder guardar las preguntas y respuestas       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


from transformers import AutoTokenizer  # cargar el tokenizador del modelo de embeddings de Hugging Face
from langchain_community.vectorstores import FAISS  # instancia para base de datos vectorial FAISS destinada para las búsquedas por similitudor similitud
from langchain_huggingface import HuggingFaceEmbeddings  # usar el modelo de embeddings de Hugging Face que convierte los chunks en vectores semánticos
import faiss  # crear y consultar la base de datos vectorial FAISS (versión CPU)
import pickle  # guardar y cargar los objetos serializados (por ejemplo, los índices)
import requests  # hacer peticiones al servidor Flask con el modelo
import logging  # controlar y personalizar la salida de mensajes, avisos y errores
import tkinter as tk  # crear la interfaz gráfica de usuario (GUI)
from tkinter import ttk, scrolledtext, filedialog, messagebox  # crear widgets, cajas de texto y diálogos de archivos

# Constantes de configuración
LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SERVER_IP = "192.168.79.82"
API_KEY = "<MASTERKEY>"
VECTOR_DB_PATH = "./vector_db"
MAX_TOKENS = 4096

# Cargar elementos de la base vectorial
with open(f"{VECTOR_DB_PATH}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")

# Cargar el modelo de embeddings de Hugging Face y cargar la base de datos FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
faiss_db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model)


# Clae para manejar la interacción del cliente con el modelo Llama3.2
class Llama3CLI:
    def __init__(self):
        self.session_id = "0"


    # Procesar la solicitud del usuario
    def process_request(self, question: str):
        # se encuentran los 5 chunks más relacionados con la pregunta dentro de la base de datos FAISS,
        # devolviéndose los objetos 'Document' correspondientes del docstore.
        docs = faiss_db.similarity_search(question, k=5)
        print(f"Chunks rescatados por similitud: {len(docs)}")
        for i, doc in enumerate(docs):
            chunk_preview = " ".join(doc.page_content.split()[:15]) + " ..."
            print(f" {i+1}. {doc.metadata['source']} (chunk {doc.metadata['chunk_index']}): {chunk_preview}")

        contexto = ""
        for doc in docs:
            contexto += doc.page_content + "\n- - - - -\n"

        prompt = (
            "Eres un asistente experto en análisis de documentos. "
            "Debes responder con precisión y claridad utilizando la información proporcionada en el siguiente contexto. "
            "Tu objetivo es entender bien la intención de la pregunta y dar una respuesta útil y coherente.\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Pregunta:\n{question}\n\n"
            "Respuesta:"
        )

        data = {
            "pooling": "none",
            "task": "generation",
            "content": [question],
            "max_tokens": MAX_TOKENS,
            "new_prompt": prompt,
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


# Clase para la interfaz gráfica de usuario (GUI) usando Tkinter
class Llama3GUI:
    def __init__(self):
        self.client = Llama3CLI()
        self.window = tk.Tk()
        self.window.title("Llama3.2 - RAG Assistant")
        self.window.geometry("900x700")
        self.window.configure(bg="#f0f2f5")
        self.build_interface()


    # Construir la interfaz gráfica
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

        # Botón de enviar
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
            command=self.send_question
        )
        self.send_button.pack(pady=(0, 5))

        # Botón para guardar en archivo
        self.save_button = tk.Button(
            main_frame,
            text="💾 Guardar pregunta y respuesta",
            font=('Segoe UI', 10),
            bg="#34a853",
            fg="white",
            activebackground="#2c8b44",
            activeforeground="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
            command=self.save_to_file
        )
        self.save_button.pack(pady=(5, 10))

        # Separador
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=10)

        # Respuesta
        label_output = ttk.Label(main_frame, text="📬 Respuesta generada:")
        label_output.pack(anchor="w")

        self.output_text = scrolledtext.ScrolledText(main_frame, height=15, font=('Segoe UI', 10), wrap=tk.WORD, bg="#ffffff")
        self.output_text.pack(fill="both", expand=True)


    # Enviar la pregunta al servidor y mostrar la respuesta
    def send_question(self):
        question = self.input_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Advertencia", "Debes escribir una pregunta antes de enviarla.")
            return

        print(f"Resolviendo a la pregunta: {question}")
        self.output_text.delete("1.0", tk.END)
        response = self.client.process_request(question)
        print(f"Consulta respondida sobre {question}")
        print("- - - - - - - - - - - - - - - - - - -")

        # se extrae el contenido de la respuesta
        content = response.get("response", "No se recibió una respuesta válida del servidor.")

        if content:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, content)
        else:
            self.output_text.insert(tk.END, "No se recibió una respuesta válida del servidor.")


    # Guardar la pregunta y respuesta en un archivo de texto
    def save_to_file(self):
        query = self.input_text.get('1.0', tk.END).strip()
        answer = self.output_text.get('1.0', tk.END).strip()

        if not query or not answer:
            messagebox.showwarning("Aviso", "No hay contenido para guardar.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")],
                                                 title="Guardar pregunta y respuesta")

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("🧾 Pregunta:\n")
                f.write(query + "\n\n\n")
                f.write("📬 Respuesta:\n")
                f.write(answer + "\n")


    # Iniciar la ventana principal de la GUI
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = Llama3GUI()
    app.run()