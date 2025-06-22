# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación Aumentada por Recuperación (RAG)      #
#           con LLaMA 3.2 como LLM para consultas                       #
#           sobre documentos o artículos en PDF                         #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: client_local1.py                                        #
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
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import logging  # controlar y personalizar la salida de mensajes, avisos y errores
import tkinter as tk  # crear la interfaz gráfica de usuario (GUI)
from tkinter import scrolledtext, filedialog, messagebox  # crear cajas de texto, personalización de archivos y widgets
from datetime import datetime  #  manejar fechas y horas
import re  # limpiar y procesar texto mediante expresiones regulares
import threading  # manejar tareas simultáneamente
import pickle  # guardar o cargar los objetos serializados (por ejemplo, los índices)
import requests  # hacer peticiones al servidor Flask que dispone del LLM
from googletrans import Translator  # traducir el texto de la pregunta siempre al inglés para una mejor interactividad con el modelo Llama3.2
import asyncio  # manejar la ejecución de código asíncrono, en este caso para la traducción


LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SERVER_IP = "192.168.XX.XX"
API_KEY = "f4d3c2b1a9876543210fedcba"
VECTOR_DB_PATH = "./vector_db"
MAX_TOKENS = 4096

# se cargan los índices de los vectores de la base de datos
index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")

# se carga el docstore de chunks y el diccionario de índices document-vector
with open(f"{VECTOR_DB_PATH}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

# se carga el modelo de embeddings, que con ello y lo anterior se abre la base de datos FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
faiss_db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model)


# Clase para manejar la interacción del cliente con el LLM (Llama3.2)
class Llama32CLI:
    def __init__(self):
        self.session_id = "0"
        self.last_docs = []
        self.last_sources = []
        self.last_timestamp = None

    # Procesar la solicitud del usuario
    def process_request(self, question: str):
        # si se hace una pregunta relativa a la misma sesión, se reutiliza el contexto de la primera pregunta
        prompt = None

        # si se trata de una nueva sesión, se encuentran los 5 chunks más relacionados de únicamente la primera query dentro de la base de datos,
        # devolviéndose los objetos 'Document' correspondientes del docstore.
        if self.session_id == "0":
            self.last_docs = faiss_db.similarity_search(question, k=6)
            print(f"Chunks rescatados por similitud: {len(self.last_docs)}")
            for i, doc in enumerate(self.last_docs):
                chunk_preview = " ".join(doc.page_content.split()[:25]) + " ..."
                print(f" {i + 1}. {doc.metadata['source']} (chunk {doc.metadata['chunk_index']}): {chunk_preview}")

            # se crea el contexto una sola vez y se almacena
            self.contexto = "\n\n".join(doc.page_content for doc in self.last_docs)
            contexto = self.contexto

            # se utiliza un prompt predefinido para enviar al LLM, que incluye lo que debe de hacer y el contexto
            prompt = (
                "Eres un asistente especializado en análisis de literatura científica y farmacéutica. "
                "Tu función es ayudar a un equipo de investigación a extraer información útil, relevante y verificable a partir del siguiente contexto, "
                "que procede de artículos académicos o técnicos.\n\n"
                f"Contexto:\n{contexto}\n\n"
                "Directrices para responder:\n"
                "- Comprende con precisión la intención de la pregunta.\n"
                "- Responde exclusivamente con la información contenida en el contexto proporcionado.\n"
                "- Si la pregunta no puede responderse con los datos disponibles, indica claramente que no hay suficiente información.\n"
                "- No inventes, asumas ni extrapoles más allá del contenido dado.\n"
                "- Utiliza un lenguaje técnico, claro y preciso, adecuado para investigadores.\n"
                "- Si procede, organiza la respuesta en secciones o puntos clave para mejorar su comprensión.\n"
            )

        docs = self.last_docs

        # se prepara el cuerpo de la petición al servidor (pooling y task controladas por la instancia Llama3.2)
        # el contenido será la query, los tokens máximos serán 4096 y el prompt el creado anteriormente
        data = {
            "content": [question],
            "max_tokens": MAX_TOKENS,
            "new_prompt": prompt,
        }

        # se configuran los headers de la petición, incluyendo la clave de API del usuario y la nueva sesión
        headers = {"Authorization": API_KEY, "Session": self.session_id}
        # se construye la URL hacia el servidor Flask que contiene el LLM
        url = f"http://{SERVER_IP}:{LLAMA_PORT}/request"

        # se envía la petición al servidor con el cuerpo y los headers
        print("Enviando el query al LLM del servidor...")
        try:
            response = requests.post(url, json=data, headers=headers)
        except Exception as ex:
            logging.error(f"Connection error: {ex}")
            return {"response": "No se pudo conectar al servidor", "status_code": "Host Unreachable"}

        # si se obtiene con éxito la respuesta, se extraen los documentos fuente y sus archivos correspondientes,
        # además de guardarse el identificador de la sesión y de retornar la respuesta en formato JSON.
        if response.status_code == 200:
            self.last_docs = docs
            self.last_sources = list({doc.metadata['source'] for doc in docs})

            resp_json = response.json()
            self.session_id = resp_json.get("session_id")
            return resp_json
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            return {"response": "Error del servidor", "status_code": response.status_code}


# Clase para crear los botones personalizados para la toma de decisiones en la interfaz Tkinter
class TkButtons(tk.Canvas):
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

        # se enlazan los eventos tanto a los elementos internos como al canvas completo
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


# Clase para definir cómo es la interfaz gráfica de usuario y controlar cualquier evento que ocurra en ella
class Llama3GUI:
    def __init__(self, root):
        self.client = Llama32CLI()
        self.root = root
        self.root.title("Chat Cliente")
        self.conversation_log = []
        self.last_docs = []
        self.last_sources = []
        self.root.bind("<Configure>", self.on_configure)
        self._last_state = self.root.state()
        self.is_processing = False
        self.was_maximized = False

        bg_color = "#f0f9f4"  # verde claro
        send_button_color = "#a8d5f2"  # azul claro
        send_button_color_hover = "#7fb7e8"  # azul más oscuro
        delete_button_color = "#f28b8b"  # rojo claro
        delete_button_color_hover = "#d9534f"  # rojo más oscuro
        save_button_color = "#a7d7c5"  # verde claro
        save_button_color_hover = "#7db8a4"  # verde más oscuro
        text_color = "#2e2e2e"  # gris oscuro
        show_button_color = "#f2e6a8" # amarillo claro
        show_button_color_hover = "#e8d17f" # amarillo más oscuro

        # se establece el color de fondo
        self.root.configure(bg=bg_color)

        # se crea un marco principal que contiene todos los elementos de la ventana.
        center_frame = tk.Frame(self.root, bg=bg_color)
        center_frame.place(relx=0.5, rely=0, relwidth=0.92, relheight=1.0, anchor="n")

        # dentro de ese marco principal, se crea un marco secundario que contiene el contenido principal.
        main_frame = tk.Frame(center_frame, bg=bg_color, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # se configura el tamaño mínimo del marco principal para que se ajuste al contenido
        label_title = tk.Label(main_frame, text="Asistente con Generación Aumentada por Recuperación - Llama 3.2", font=("Inter", 18, "bold"), bg=bg_color, fg=text_color)
        label_title.pack(pady=(0, 10))

        # se añade el texto donde se indica al usuario que escriba su pregunta
        label_input = tk.Label(main_frame, text=" Pregunta:", font=("Segoe UI", 13, "italic"), bg=bg_color, fg=text_color)
        label_input.pack(anchor="w")

        # se crea el campo de texto para que el usuario pueda escribir su pregunta
        self.input_text = tk.Text(main_frame, height=3, font=('Segoe UI', 11), wrap=tk.WORD, bg="white", fg=text_color, insertbackground=text_color)
        self.input_text.pack(fill="x", pady=(5, 10))

        # se crea el marco que contendrá el botón de enviar justo debajo de la anterior ventana.
        btn_frame = tk.Frame(main_frame, bg=bg_color)
        btn_frame.pack(fill="x", pady=(0, 10))

        # dentro de ese marco para el botón, se reserva espacio para ubicarlo.
        send_frame = tk.Frame(btn_frame, bg=bg_color)
        send_frame.pack(side="left", expand=True)

        # se incorpora el botón 'Enviar'
        self.send_button = TkButtons(send_frame, "Enviar", command=self.send_question, bg=send_button_color, fg=text_color, hover_bg=send_button_color_hover, width=140, height=60, radius=15, font=("Segoe UI", 14, "bold"))
        self.send_button.pack()

        # se añade el texto donde se indica al usuario que escriba su pregunta
        label_output = tk.Label(main_frame, text=" Historial de la conversación:", font=("Segoe UI", 13, "italic"), bg=bg_color, fg=text_color)
        label_output.pack(anchor="w")

        # se crea el campo de texto donde se mostrará el historial de la conversación scrolleable
        self.chat_text = scrolledtext.ScrolledText(main_frame, height=20, font=('Segoe UI', 10), wrap=tk.WORD, state="disabled", bg="white", fg=text_color, insertbackground=text_color)
        self.chat_text.pack(fill="both", expand=True, pady=(5, 10))

        # se configuran los estilos para los contenidos del chat
        self.chat_text.tag_configure("timestamp", foreground="#888888", font=("Segoe UI", 9, "italic"), justify="center", spacing3=6)
        self.chat_text.tag_configure("label_user", foreground="#5387bf", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("bold_user", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("label_assistant", foreground="#4a7c59", font=("Segoe UI", 11, "bold"))
        self.chat_text.tag_configure("user", justify="left", background="#e6f4ff", font=("Segoe UI", 11, "bold"), lmargin1=10, lmargin2=10, rmargin=150, spacing3=4)
        self.chat_text.tag_configure("assistant", justify="right", font=("Segoe UI", 11), lmargin1=150, lmargin2=10, rmargin=10, spacing3=4)
        self.chat_text.tag_configure("thinking", justify="right", font=("Segoe UI", 11, "italic"), lmargin1=150, lmargin2=10, rmargin=10, spacing3=4)

        # se crea un marco para los botones restantes justo debajo de la anterior ventana
        bottom_button_frame = tk.Frame(main_frame, bg=bg_color)
        bottom_button_frame.pack(pady=(0, 10))

        # se incorporan los botones 'Reiniciar conversación' y 'Guardar conversación'
        self.sources_button = TkButtons(bottom_button_frame,"Comprobar archivos fuente", command=self.show_sources, bg=show_button_color, fg=text_color, hover_bg=show_button_color_hover, width=230, height=45, radius=15, font=("Segoe UI", 12, "bold"))
        self.sources_button.pack(side="left")
        self.clear_button = TkButtons(bottom_button_frame, "Reiniciar conversación", command=self.restart_chat, bg=delete_button_color, fg=text_color, hover_bg=delete_button_color_hover, width=200, height=45, radius=15, font=("Segoe UI", 12, "bold"))
        self.clear_button.pack(side="left", padx=120)
        self.save_button = TkButtons(bottom_button_frame, "Guardar conversación", command=self.save_conversation, bg=save_button_color, fg=text_color, hover_bg=save_button_color_hover, width=200, height=45, radius=15, font=("Segoe UI", 12, "bold"))
        self.save_button.pack(side="left", ipadx=15)

    # Redimensionar la ventana a más pequeña
    def on_configure(self, event):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width, height = self.root.winfo_width(), self.root.winfo_height()

        is_now_small = width < screen_width - 50 and height < screen_height - 50

        if self.was_maximized and is_now_small:
            self.fit_window_to_contents()

        self.was_maximized = width >= screen_width - 20 and height >= screen_height - 80

    # Ajustar el tamaño de la ventana al contenido actual
    def fit_window_to_contents(self):
        self.root.update_idletasks()

        content_frame = self.root.winfo_children()[0]
        content_width = content_frame.winfo_reqwidth()
        content_height = content_frame.winfo_reqheight()

        extra_width = 60
        extra_height = 100

        new_width = content_width + extra_width
        new_height = content_height + extra_height

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        final_width = min(new_width, screen_width - 40)
        final_height = min(new_height, screen_height - 80)

        x = (screen_width - final_width) // 2
        y = (screen_height - final_height) // 2

        self.root.geometry(f"{final_width}x{final_height}+{x}+{y}")


    # Enviar la pregunta al servidor y procesar la respuesta
    def send_question(self):
        if self.is_processing:
            messagebox.showwarning("Advertencia", "Por favor, espere a que la última respuesta sea respondida antes de enviar otra pregunta.")
            return

        question = self.input_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Advertencia", "Debe escribir una pregunta antes de enviarla.")
            return

        # se activa la variable de "proceso en curso" y se acomoda el cursor al estado inicial
        self.is_processing = True
        self.root.config(cursor="")

        # se almacena la hora y fecha de la pregunta realizada y se elimina la pregunta en el widget de arriba
        timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.client.last_timestamp = timestamp
        self.input_text.delete("1.0", tk.END)

        # se muestra la pregunta y el "Pensando...", además de marcar ese mensaje para sustituirlo por la respuesta posteriormente
        self.chat_text.config(state="normal")
        self.chat_text.insert(tk.END, f"{timestamp}\n", "timestamp")
        self.chat_text.insert(tk.END, " Tú: ", "label_user")
        self.chat_text.insert(tk.END, f"{question}\n\n", "bold_user")
        self.chat_text.insert(tk.END, "Pensando...\n\n", ("thinking",))
        self.chat_text.tag_add("thinking_tag", "end-3l", "end-1l")
        self.chat_text.config(state="disabled")

        # se desplaza el scrolltext hasta el último mensaje
        self.chat_text.yview(tk.END)
        # se fuerza la actualización de la interfaz para ver los mensajes antes de tareas largas
        self.root.update_idletasks()

        # se crea un hilo "ficticio" para procesar la pregunta y devolver una respuesta por separado,
        # consiguiendo así sustituir el "Pensando..." con la respuesta y no bloquear la GUI
        threading.Thread(target=self.process_question_in_thread, args=(question,)).start()

    # Procesar la pregunta en el hilo
    def process_question_in_thread(self, question):
        # en primer lugar, se traduce la pregunta al inglés, por si acaso
        translator = Translator()
        tr_question = asyncio.run(translator.translate(question, dest='en')).text

        # se llama a la función que realiza la petición al servidor que dispone del LLM para formar la respuesta
        response = self.client.process_request(tr_question)
        self.last_sources = self.client.last_sources
        self.last_docs = self.client.last_docs

        # se usa la función que muestra la respuesta en el hilo principal usando '.after'
        self.root.after(0, self.display_response, response, question)

    def display_response(self, response, question):
        self.chat_text.config(state="normal")

        # se consigue el contenido de la respuesta comprobándose el éxito de la misma
        if isinstance(response, dict) and response.get("status_code") == 200:
            content = response.get("response", {})
            if isinstance(content, dict) and "content" in content:
                content = content["content"]
            else:
                content = str(content)

            # se borra lo que haya en el intervalo del mensaje "Pensando..." y se borra la propia etiqueta que lo ubica allí
            self.chat_text.delete("thinking_tag.first", "thinking_tag.last")
            self.chat_text.tag_delete("thinking_tag")

            # se expone la respuesta respetando posibles partes en negrita
            self.chat_text.insert(tk.END, "Asistente: ", "label_assistant")
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
        # se desplaza el scrolltext hasta el último mensaje
        self.chat_text.yview(tk.END)

        # se guarda la fecha de la pregunta, la propia pregunta, la respuesta y los archivos fuente utilizados
        self.conversation_log.append({
            "timestamp": self.client.last_timestamp,
            "question": question,
            "answer": content.strip(),
            "docs": self.last_docs or []
        })

        # se desactiva la variable "proceso en curso"
        self.is_processing = False


    # Enseñar los archivos fuente utilizados en las respuestas de la conversación
    def show_sources(self):
        if not self.conversation_log:
            messagebox.showwarning("Advertencia", "Todavía no se ha realizado ninguna pregunta.")
            return

        # se cogen los objetos "Document" de los archivos fuente
        docs = self.conversation_log[0].get("docs", [])

        # se crea la ventana emergente para mostrar las fuentes
        top = tk.Toplevel(self.root)
        top.title("Fuentes de las respuestas de la conversación")
        top.configure(bg="#f0f9f4")

        # se establece el tamaño y la ubicación de la ventana
        width, height = 640, 420
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        top.geometry(f"{width}x{height}+{x}+{y}")

        # se crea el cuadro de texto scrolleable que dispone de la información
        text_area = scrolledtext.ScrolledText(top, wrap=tk.WORD, font=("Segoe UI", 10), bg="white", fg="#2e2e2e")
        text_area.pack(expand=True, fill="both", padx=10, pady=10)

        # se establece el primer mensaje informativo y de seguido esos archivos fuente
        text_area.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        text_area.insert(tk.END, "Fuentes utilizadas para responder a las preguntas:\n\n", "bold")
        for i, doc in enumerate(docs):
            source_path = os.path.normpath(doc.metadata.get("source"))
            chunk_index = doc.metadata.get("chunk_index")
            preview = " ".join(doc.page_content.split()[:25]) + " ..."
            text_area.insert(tk.END, f"{i + 1}. {source_path} (chunk {chunk_index}):\n\n{preview}\n\n\n\n")

        text_area.config(state="disabled")

    # Reiniciar la conversación y limpiar el historial
    def restart_chat(self):
        # se evita interrumpir una respuesta en curso
        if self.is_processing:
            messagebox.showwarning("Advertencia", "Por favor, espere a que la última respuesta sea respondida antes de reiniciar la conversación.")
            return

        # se evita reiniciar si no hay conversación previa
        current_text = self.chat_text.get("1.0", tk.END).strip()
        if not current_text:
            messagebox.showwarning("Advertencia", "No hay conversación que reiniciar."
            )
            return

        # se comprueba si el usuario realmente quiere reiniciar la conversación
        confirm = messagebox.askokcancel("Confirmación de reiniciado de conversación", "¿Está seguro de que desea generar una nueva conversación? "
                                                                                                 "Se perderá el contexto y el historial actual.")
        if not confirm:
            return

        # se elimina el contenido del campo de texto del chat y se limpia el historial de conversación
        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state="disabled")
        self.conversation_log.clear()

        # se reinicia la identificación de sesión actual del cliente para que se genere una conversación de cero, sin mantener el contexto anterior
        # además, se reinician los documentos del contexto para la conversación
        self.client.session_id = "0"
        self.client.last_docs = []
        self.client.contexto = None


    # Guardar la conversación en un archivo de texto
    def save_conversation(self):
        # se evita guardar si no hay conversación previa
        if not self.conversation_log:
            messagebox.showwarning("Advertencia", "No hay conversación para guardar.")
            return

        # se guarda personalizadamente en un archivo de texto
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title=f"{self.conversation_log[0]['timestamp']}" )
        if not filepath:
            return

        # se abre el archivo para escribir el contenido de la conversación y los archivos de apoyo
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # se obtienen los archivos fuente del primer mensaje (PDF con su chunk correspondiente)
                if self.conversation_log:
                    docs = self.conversation_log[0].get("docs", [])
                    if docs:
                        f.write("Fuentes utilizadas en la conversación:\n\n")
                        for i, doc in enumerate(docs):
                            source_path = os.path.normpath(doc.metadata.get("source", "desconocido"))
                            chunk_preview = " ".join(doc.page_content.split()[:25]) + " ..."
                            f.write(f"  {i + 1}. {source_path}: {chunk_preview}\n")
                        f.write("\n\n")

                # se escribe el historial de la conversación
                for entry in self.conversation_log:
                    f.write(f"[{entry['timestamp']}]\n\n")
                    f.write(f"Tú: {entry['question']}\n\n")
                    f.write(f"Asistente: {entry['answer']}\n\n\n")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la conversación:\n{e}")


# Función principal
if __name__ == "__main__":
    # se crea la instancia principal de la ventana de la aplicación mediante Tkinter
    root = tk.Tk()
    # se establece el estado inicial de la ventana como "zoomed" para que ocupe toda la pantalla
    root.state("zoomed")
    # se fuerza el enfoque de la ventana principal
    root.focus_force()
    # se inicializa la aplicación principal, pasando la ventana raíz como argumento.
    app = Llama3GUI(root)
    # se inicia el bucle principal de la aplicación, que mantiene la ventana abierta y responde a eventos.
    root.mainloop()