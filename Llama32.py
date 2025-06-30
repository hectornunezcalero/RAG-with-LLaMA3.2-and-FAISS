# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática - Curso 2025/2026                #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación Aumentada por Recuperación (RAG)      #
#           con LLaMA 3.2 como LLM para consultas                       #
#           sobre documentos o artículos en PDF                         #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Tutor: Jorge Pérez Aracil                                       #
#       Cotutor: Alberto Palomo Alonso                                  #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: Llama32.py                                              #
#       Funciones principales:                                          #
#        1. Inicializar el modelo LLaMa 3.2 3B localmente               #
#        2. Utilizar el tokenizador y pipeline de transformers          #
#        3. Gestionar mensajes system, usuario y asistente              #
#        4. Ejecutar la generación de texto y devolver respuesta        #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

import torch  # manejar el modelo en GPU o CPU
from transformers import pipeline, AutoTokenizer  # cargar el pipeline y tokenizador del modelo de embeddings de Hugging Face
import logging  # controlar y personalizar la salida de mensajes, avisos y errores

# se establece el número máximo de mensajes que se pueden manejar en una conversación (1 prompt + 10 preguntas + 10 respuestas)
MAX_MESSAGES = 21

# se establece la ruta del modelo Llama3.2
__llama_path__ = "./model"


# Clase del modelo Llama3.2 que maneja las tareas de NLP
class Llama3:
    def __init__(self, max_tokens: int = 4096, pooling: str = 'none', gpu: bool = True):
        # se inicializan los atributos del modelo:
        self.pipe = None
        self.tokenizer = None
        self.task = 'generation'
        self.messages: list = list()
        self.prompt: dict = {}
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.gpu = gpu

        # se prepara el pipeline del modelo Llama3.2
        self._init_pipeline()

    # Establecer el pipeline del modelo
    def _init_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(__llama_path__)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # el nuevo pipeline se crea con el modelo y el tokenizador adecuados
        self.pipe = pipeline(
            "text-generation",
            tokenizer = self.tokenizer,
            model = __llama_path__,
            torch_dtype = torch.bfloat16,
            device_map = 0 if self.gpu else 'cpu',
        )
        logging.info("Modelo Llama3.2 cargado para generación de texto.")


    # Establecer el prompt inicial del modelo para comenzar una conversación
    def set_prompt(self, prompt: str):
        # el rol de esta transacción es "system" y el contenido el prompt
        self.prompt = {'role': 'system', 'content': prompt}

        # se inicializa la lista de mensajes con el prompt como primer mensaje
        self.messages = [self.prompt]
        logging.info(f'Prompt establecido: {prompt}')


    # Gestionar la consulta recibida, para que el modelo pueda responderla
    def text_generation_task(self, text: list[str], max_tokens):
        # el rol de esta transacción es "user" y el contenido la pregunta recibida
        self.messages.append({'role': 'user', 'content': text})

        # si el modelo ya tiene mensajes, para no perder funcionamiento,
        # se eliminan mensajes menos el prompt y los últimos mensajes pregunta-respuesta
        if len(self.messages) > MAX_MESSAGES:
            self.messages = [self.prompt] + self.messages[-(MAX_MESSAGES - 1):]

        # se genera la respuesta del modelo sobre la pregunta recibida
        response = self.pipe(messages=self.messages, max_new_tokens=max_tokens)[0]
        generated_text = response["generated_text"].strip()

        # se añade el mensaje de la respuesta del modelo a la lista de mensajes y se devuelve dicha respuesta
        self.messages.append({'role': 'assistant', 'content': generated_text})
        return generated_text


    # Permitir que la instancia Llama3 se use como función en el servidor
    def __call__(self, *args, **kwargs):
        # se obtiene el número máximo de tokens de los argumentos
        max_tokens = kwargs.get("tokens", self.max_tokens)
        # se llama directamente a la tarea de generación de texto con los argumentos recibidos
        return self.text_generation_task(list(args), max_tokens)

    # Devolver una representación legible de la instancia Llama3
    def __repr__(self):
        return f'<Llama3.2: (tarea={self.task}, max_tokens={self.max_tokens})>'