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
#        1. Inicializar el modelo LLaMA 3.2 3B localmente               #
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
__llama_path__ = "../model"


class Llama3:
    """
    Class to manage the Llama3.2 text generation model based on Hugging Face Transformers.
    Provides functionality to set an initial prompt, manage conversational context,
    and generate text responses to user input.
    Attributes:
        pipe (pipeline): Hugging Face pipeline for text generation.
        tokenizer (AutoTokenizer): Tokenizer associated with the model.
        task (str): Task type (default 'generation').
        messages (list): List of messages maintaining conversational context.
        prompt (dict): Initial prompt of the conversation.
        max_tokens (int): Maximum number of tokens to generate in responses.
        pooling (str): Pooling strategy (currently unused).
        gpu (bool): Whether to run the model on GPU (True) or CPU (False).
    """
    def __init__(self, max_tokens: int = 1024, pooling: str = 'none', gpu: bool = True):
        """
        Initializes the model instance, configuring pipeline, tokenizer, and parameters.
        Args:
            max_tokens (int): Maximum tokens allowed in generated text.
            pooling (str): Pooling strategy for embeddings (optional).
            gpu (bool): Flag to use GPU (True) or CPU (False).
        """
        self.pipe = None
        self.tokenizer = None
        self.task = 'generation'
        self.messages: list = list()
        # inicializar prompt por defecto para evitar KeyError si no se llama a set_prompt()
        self.prompt: dict = {'role': 'system', 'content': ''}
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.gpu = gpu

        # se prepara el pipeline del modelo Llama3.2
        self._init_pipeline()

    def _init_pipeline(self):
        """
        Initializes the text generation pipeline and tokenizer for Llama3.2.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(__llama_path__)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # el nuevo pipeline se crea con el tokenizador, modelo, formato de tensores y unidad para llevarlo a cabo
        self.pipe = pipeline(
            "text-generation",
            tokenizer = self.tokenizer,
            model = __llama_path__,
            torch_dtype = torch.bfloat16, # float32 para mayor eficiencia en GPU
            device_map = 0, # 0 para usar la GPU si está disponible, 'cpu' para usar la CPU
        )
        logging.info("Modelo Llama3.2 cargado para generación de texto.")


    def set_prompt(self, prompt: str):
        """
        Sets the initial prompt to start a conversation with the model.
        Args:
            prompt (str): Text defining the initial or system context for the model.
        """
        # el rol de esta transacción es "system" y el contenido el prompt
        self.prompt = {'role': 'system', 'content': prompt}

        # se inicializa la lista de mensajes con el prompt como primer mensaje
        self.messages = [self.prompt]
        logging.info(f'Prompt establecido: {prompt}')


    def text_generation_task(self, text, max_tokens: int):
        """
        Handles a text query and generates a response from the model.
        Args:
            text (list[str]): List of input user messages.
            max_tokens (int): Maximum number of tokens to generate in the response.
        Returns:
            str: Generated text response from the model.
        """
        # normalizar entrada: aceptar str o lista/tupla de strings
        if isinstance(text, (list, tuple)):
            user_content = " ".join(map(str, text))
        else:
            user_content = str(text)

        # asegurar que prompt y messages están inicializados correctamente
        if not isinstance(self.prompt, dict) or 'role' not in self.prompt or 'content' not in self.prompt:
            self.prompt = {'role': 'system', 'content': ''}
        if not self.messages:
            self.messages = [self.prompt]

        # añadir mensaje de usuario
        self.messages.append({'role': 'user', 'content': user_content})

        # podar conversación si excede el máximo
        if len(self.messages) > MAX_MESSAGES:
            self.messages = [self.prompt] + self.messages[-20:]

        # construir prompt acumulado de forma segura
        parts = []
        for m in self.messages:
            role = m.get('role', 'unknown')
            content = m.get('content', '')
            parts.append(f"{role}: {content}")
        acc_prompt = "\n".join(parts)

        # comprobar pipeline
        if self.pipe is None:
            logging.error('Pipeline no inicializado antes de generar texto')
            raise RuntimeError('Pipeline no inicializado')

        # generar respuesta
        response = self.pipe(acc_prompt, max_new_tokens=max_tokens)[0]
        generated_text = response.get('generated_text', '').strip()

        # recortar prompt si el modelo lo antepuso
        if generated_text.startswith(acc_prompt):
            generated_text = generated_text[len(acc_prompt):].strip()

        # almacenar la respuesta y devolverla
        self.messages.append({'role': 'assistant', 'content': generated_text})
        return generated_text


    def __call__(self, *args, **kwargs):
        """
        Enables the Llama3 instance to be called as a function for direct text generation.
        Args:
            *args: Input text arguments.
            **kwargs: Additional parameters such as 'tokens' for max tokens.
        Returns:
            str: Generated text response from the model.
        """
        # se obtiene el número máximo de tokens de los argumentos
        max_tokens = kwargs.get("tokens", self.max_tokens)

        # normalizar los argumentos posicionales a una única cadena
        if not args:
            text = ""
        elif len(args) == 1:
            text = args[0]
        else:
            text = " ".join(map(str, args))

        # llamar a la función principal de generación
        return self.text_generation_task(text, max_tokens)

    def __repr__(self):
        """
        Returns a readable representation of the Llama3 instance.
        Returns:
            str: Description of the Llama3 object with task and max_tokens.
        """
        return f'<Llama3.2: (tarea={self.task}, max_tokens={self.max_tokens})>'

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#                               END OF FILE                             #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #