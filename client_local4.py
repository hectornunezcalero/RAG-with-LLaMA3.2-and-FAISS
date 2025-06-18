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
#           sobre documentos PDF                                        #
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
#        1. Prestar la GUI con Tkinter para interactuar con LLaMa 3.2   #
#           a modo de pregunta-respuesta                                #
#        2. Buscar documentos relacionados en la base de datos FAISS    #
#        3. Enviar consultas al servidor que dispone del LLM            #
#        4. Visualizar y poder guardar las preguntas y respuestas       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

from transformers import AutoTokenizer  # cargar el tokenizador del modelo de embeddings de Hugging Face
from langchain_community.vectorstores import FAISS  # instancia para base de datos vectorial FAISS destinada para las búsquedas por similitud
from langchain_huggingface import HuggingFaceEmbeddings  # usar el modelo de embeddings de Hugging Face que convierte los chunks en vectores semánticos
import faiss  # crear y consultar la base de datos vectorial FAISS (versión CPU)
import pickle  # guardar y cargar los objetos serializados (por ejemplo, los índices)
import requests  # hacer peticiones al servidor Flask con el modelo
import logging  # controlar y personalizar la salida de mensajes, avisos y errores
import tkinter as tk  # crear la interfaz gráfica de usuario (GUI)
from tkinter import scrolledtext, filedialog, messagebox  # crear widgets, cajas de texto y diálogos de archivos
from googletrans import Translator  # traducir el texto de la pregunta al inglés para el modelo Llama3.2
import asyncio  # manejar corutinas
from datetime import datetime  #  manejar fechas y horas
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import re  # manejar expresiones regulares para formatear el texto de la respuesta
import threading  # manejar tareas simultáneamente

LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SERVER_IP = "192.168.XX.XX"
API_KEY = "7f6e5d4c3b2a1098f7e6d5c4b"
VECTOR_DB_PATH = "./vector_db"
MAX_TOKENS = 4096

# Cargar elementos de la base vectorial
with open(f"{VECTOR_DB_PATH}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")

# Cargar el modelo de embeddings de Hugging Face y cargar la base de datos FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
faiss_db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model)


# Clase para manejar la interacción del cliente con el modelo Llama3.2
class Llama3CLI:
    def __init__(self):
        self.session_id = "0"
        self.last_sources = []
        self.last_docs = []
        self.last_timestamp = None

    # Procesar la solicitud del usuario
    def process_request(self, question: str):
        # se encuentran los 5 chunks más relacionados con la pregunta dentro de la base de datos FAISS,
        # devolviéndose los objetos 'Document' correspondientes del docstore.
        docs = faiss_db.similarity_search(question, k=6)
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
            self.last_sources = list({doc.metadata['source'] for doc in docs})
            self.last_docs = docs

            resp_json = response.json()
            self.session_id = resp_json.get("session_id", "0")
            return resp_json
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            return {"response": "Error del servidor", "status_code": response.status_code}


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, font=None, bg="#a7d7c5", fg="#2e2e2e", hover_bg="#c5e1d8", width=140, height=50, radius=20):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent['bg'])
        self.command = command
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg
        self.radius = radius
        self.width = width
        self.height = height
        self.font = font or ("Segoe UI", 10, "bold")

        self.border_width = 2
        self.border_color = "#4a4a4a"

        self.text = text

        self._draw_button(self.bg)

        # Enlazar eventos SOLO a los elementos internos
        self.tag_bind("button_shape", "<Enter>", self.on_enter)
        self.tag_bind("button_text", "<Enter>", self.on_enter)
        self.tag_bind("button_shape", "<Leave>", self.on_leave)
        self.tag_bind("button_text", "<Leave>", self.on_leave)
        self.tag_bind("button_shape", "<Button-1>", self.on_click)
        self.tag_bind("button_text", "<Button-1>", self.on_click)

    def _draw_button(self, fill_color):
        self.delete("all")
        r = self.radius
        bw = self.border_width
        w = self.width
        h = self.height

        points = [
            bw+r, bw,
            w-bw-r, bw,
            w-bw, bw,
            w-bw, bw+r,
            w-bw, h-bw-r,
            w-bw, h-bw,
            w-bw-r, h-bw,
            bw+r, h-bw,
            bw, h-bw,
            bw, h-bw-r,
            bw, bw+r,
            bw, bw
        ]

        self.round_rect = self.create_polygon(points, smooth=True, splinesteps=36, fill=fill_color, outline=self.border_color, width=bw, tags="button_shape")

        self.text_id = self.create_text(w // 2, h // 2, text=self.text, fill=self.fg, font=self.font, tags="button_text")

    def on_enter(self, event):
        self.itemconfig(self.round_rect, fill=self.hover_bg)
        self.config(cursor="hand2")

    def on_leave(self, event):
        self.itemconfig(self.round_rect, fill=self.bg)
        self.config(cursor="arrow")

    def on_click(self, event):
        if self.command:
            self.command()


class Llama3GUI:
    def __init__(self, root):
        self.client = Llama3CLI()
        self.root = root
        self.root.title("Chat Cliente")
        self.conversation_log = []
        self.last_sources = []
        self.last_docs = []
        self.root.bind("<Configure>", self.on_configure)
        self._last_state = self.root.state()
        self.is_processing = False

        bg_color = "#f0f9f4"
        button_color_1 = "#a8d5f2"
        button_hover_1 = "#7fb7e8"
        button_color_2 = "#f28b8b"
        button_hover_2 = "#d9534f"
        button_color_3 = "#a7d7c5"
        button_hover_3 = "#7db8a4"
        text_color = "#2e2e2e"

        self.root.configure(bg=bg_color)

        center_frame = tk.Frame(self.root, bg=bg_color)
        center_frame.place(relx=0.5, rely=0, relwidth=0.92, relheight=1.0, anchor="n")

        main_frame = tk.Frame(center_frame, bg=bg_color, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        label_title = tk.Label(main_frame, text="Asistente con Generación Aumentada por Recuperación - Llama 3.2", font=("Inter", 18, "bold"), bg=bg_color, fg=text_color)
        label_title.pack(pady=(0, 10))

        label_input = tk.Label(main_frame, text=" Pregunta:", font=("Segoe UI", 12, "italic"), bg=bg_color, fg=text_color)
        label_input.pack(anchor="w")

        self.input_text = tk.Text(main_frame, height=3, font=('Segoe UI', 10), wrap=tk.WORD, bg="white", fg=text_color, insertbackground=text_color)
        self.input_text.pack(fill="x", pady=(5, 10))

        btn_frame = tk.Frame(main_frame, bg=bg_color)
        btn_frame.pack(fill="x", pady=(0, 10))

        send_frame = tk.Frame(btn_frame, bg=bg_color)
        send_frame.pack(side="left", expand=True)

        # Botón Enviar centrado a la izquierda
        self.send_button = RoundedButton(send_frame, "Enviar", command=self.send_question, bg=button_color_1, fg=text_color, hover_bg=button_hover_1, width=140, height=60, radius=15, font=("Segoe UI", 14, "bold"))
        self.send_button.pack()

        label_output = tk.Label(main_frame, text=" Historial de la conversación:", font=("Segoe UI", 12, "italic"), bg=bg_color, fg=text_color)
        label_output.pack(anchor="w")

        self.chat_text = scrolledtext.ScrolledText(main_frame, height=20, font=('Segoe UI', 10), wrap=tk.WORD, state="disabled", bg="white", fg=text_color, insertbackground=text_color)
        self.chat_text.pack(fill="both", expand=True, pady=(5, 10))

        bottom_button_frame = tk.Frame(main_frame, bg=bg_color)
        bottom_button_frame.pack(pady=(0, 10))

        self.clear_button = RoundedButton(bottom_button_frame, "Reiniciar conversación", command=self.clear_chat, bg=button_color_2, fg=text_color, hover_bg=button_hover_2, width=200, height=45, radius=15, font=("Segoe UI", 12, "bold"))

        self.save_button = RoundedButton(bottom_button_frame, "Guardar conversación", command=self.save_conversation, bg=button_color_3, fg=text_color, hover_bg=button_hover_3, width=200, height=45, radius=15, font=("Segoe UI", 12, "bold"))

        self.clear_button.pack(side="left", padx=50)
        self.save_button.pack(padx=50, side="left")

        # Estilos
        self.chat_text.tag_configure("timestamp", foreground="#888888", font=("Segoe UI", 9, "italic"), justify="center", spacing3=6)
        self.chat_text.tag_configure("user_label", foreground="#5387bf", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("bold_user", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("assistant_label", foreground="#4a7c59", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("user", justify="left", background="#e6f4ff", font=("Segoe UI", 11, "bold"), lmargin1=10, lmargin2=10, rmargin=150, spacing3=4)
        self.chat_text.tag_configure("assistant", justify="right", font=("Segoe UI", 11), lmargin1=150, lmargin2=10, rmargin=10, spacing3=4)
        self.chat_text.tag_configure("thinking", justify="right", font=("Segoe UI", 11, "italic"), lmargin1=150, lmargin2=10, rmargin=10, spacing3=4)

    def on_configure(self, event):
        # Detecta si se redujo de tamaño desde maximizado (tamaño igual a pantalla menos bordes)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Se considera maximizada si casi llena toda la pantalla
        was_maximized = getattr(self, "_was_maximized", False)
        width, height = self.root.winfo_width(), self.root.winfo_height()

        is_now_small = width < screen_width - 50 and height < screen_height - 50

        if was_maximized and is_now_small:
            self.fit_window_to_contents()

        # Actualiza el estado
        self._was_maximized = width >= screen_width - 20 and height >= screen_height - 80


    def fit_window_to_contents(self):
        self.root.update_idletasks()

        # Obtenemos el frame que contiene el contenido real
        content_frame = self.root.winfo_children()[0]  # center_frame
        content_width = content_frame.winfo_reqwidth()
        content_height = content_frame.winfo_reqheight()

        # Márgenes de seguridad visual
        extra_width = 60   # ancho adicional
        extra_height = 100  # alto adicional para bordes + títulos

        new_width = content_width + extra_width
        new_height = content_height + extra_height

        # Límites máximos (pantalla)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        final_width = min(new_width, screen_width - 40)
        final_height = min(new_height, screen_height - 80)

        # Centramos la ventana
        x = (screen_width - final_width) // 2
        y = (screen_height - final_height) // 2

        self.root.geometry(f"{final_width}x{final_height}+{x}+{y}")


    def clear_chat(self):
        if self.is_processing:
            messagebox.showwarning("Advertencia", "Por favor, espere a que la última respuesta sea respondida antes de reiniciar la conversación.")
            return

        current_text = self.chat_text.get("1.0", tk.END).strip()
        if not current_text:
            messagebox.showwarning(
                title="Advertencia",
                message="No hay conversación que reiniciar."
            )
            return

        confirm = messagebox.askokcancel(
            title="Confirmación de reiniciado de conversación",
            message="¿Está seguro de que desea generar una nueva conversación? "
                    "Se perderá el contexto y el historial actual."
        )
        if not confirm:
            return

        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state="disabled")
        self.conversation_log.clear()

        # se reinicia la 'id' de sesión del cliente para que se genere una conversación de cero
        self.client.session_id = "0"


    def send_question(self):
        if self.is_processing:
            messagebox.showwarning("Advertencia", "Por favor, espere a que la última respuesta sea respondida antes de enviar otra pregunta.")
            return

        question = self.input_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Advertencia", "Debe escribir una pregunta antes de enviarla.")
            return

        self.is_processing = True

        self.root.config(cursor="")

        # Mostrar pregunta y "Pensando..." YA en el hilo principal para que se vea rápido
        timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.client.last_timestamp = timestamp
        self.input_text.delete("1.0", tk.END)

        self.chat_text.config(state="normal")
        self.chat_text.insert(tk.END, f"{timestamp}\n", "timestamp")
        self.chat_text.insert(tk.END, " Tú: ", "user_label")
        self.chat_text.insert(tk.END, f"{question}\n\n", "bold_user")
        self.chat_text.insert(tk.END, "Pensando...\n\n", ("thinking",))
        self.chat_text.tag_add("thinking_tag", "end-3l", "end-1l")
        self.chat_text.config(state="disabled")
        self.chat_text.yview(tk.END)
        self.root.update_idletasks()

        # Crear un hilo para procesar la pregunta y la respuesta
        threading.Thread(target=self.process_question_in_thread, args=(question,)).start()

    def process_question_in_thread(self, question):
        # Traducción (sin try/except)
        translator = Translator()
        tr_question = asyncio.run(translator.translate(question, dest='en')).text

        # Petición al servidor (sin try/except)
        response = self.client.process_request(tr_question)
        self.last_sources = self.client.last_sources
        self.last_docs = self.client.last_docs

        # Mostrar respuesta en el hilo principal usando .after
        self.root.after(0, self.display_response, response, question)

        # Volver cursor normal en hilo principal
        self.root.after(0, lambda: self.root.config(cursor=""))

    def display_response(self, response, question):
        self.chat_text.config(state="normal")

        if isinstance(response, dict) and response.get("status_code") == 200:
            content = response.get("response", {})
            if isinstance(content, dict) and "content" in content:
                content = content["content"]
            else:
                content = str(content)

            # Borrar "Pensando..."
            self.chat_text.delete("thinking_tag.first", "thinking_tag.last")
            self.chat_text.tag_delete("thinking_tag")

            self.chat_text.insert(tk.END, "Asistente: ", "assistant_label")
            parts = re.split(r"(\*\*.*?\*\*)", content.strip())
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    self.chat_text.insert(tk.END, part[2:-2], ("assistant", "bold"))
                else:
                    self.chat_text.insert(tk.END, part, "assistant")
            self.chat_text.insert(tk.END, "\n\n", "assistant")

        else:
            content = "Error: No se recibió una respuesta válida del servidor."
            self.chat_text.delete("thinking_tag.first", "thinking_tag.last")
            self.chat_text.tag_delete("thinking_tag")
            self.chat_text.insert(tk.END, f"{content}\n\n", "assistant")

        self.chat_text.config(state="disabled")
        self.chat_text.yview(tk.END)

        # Guardar en log
        timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.conversation_log.append({
            "timestamp": timestamp,
            "question": question,
            "answer": content.strip(),
            "docs": self.last_docs or []
        })

        self.is_processing = False


    def save_conversation(self):
        if not self.conversation_log:
            messagebox.showwarning("Advertencia", "No hay conversación para guardar.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="💾 Guardado de conversación"
        )
        if not filepath:
            return  # Cancelado por el usuario

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for entry in self.conversation_log:
                    f.write(f"[{entry['timestamp']}]\n\n")
                    f.write(f"Tú: {entry['question']}\n\n")
                    f.write(f"Asistente: {entry['answer']}\n\n")

                    docs = entry.get("docs", [])
                    if docs:
                        f.write("Fuentes:\n")
                        for i, doc in enumerate(docs):
                            source_path = os.path.normpath(doc.metadata.get("source", "desconocido"))
                            chunk_preview = " ".join(doc.page_content.split()[:15]) + " ..."
                            f.write(f"  {i + 1}. {source_path}: {chunk_preview}\n")
                    f.write("\n\n")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la conversación:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    root.focus_force()

    app = Llama3GUI(root)
    root.mainloop()