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
from transformers import AutoTokenizer, AutoModelForCausalLM  # cargar el pipeline y tokenizador del modelo de embeddings de Hugging Face
import logging  # controlar y personalizar la salida de mensajes, avisos y errores

# se establece el número máximo de mensajes que se pueden manejar en una conversación (1 prompt + 10 preguntas + 10 respuestas)
MAX_MESSAGES = 21
MAX_HISTORY = MAX_MESSAGES - 1

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
        self.model = None
        self.tokenizer = None
        self.task = 'generation'
        self.messages: list = list()
        # inicializar prompt por defecto para evitar KeyError si no se llama a set_prompt()
        self.prompt: dict = {'role': 'system', 'content': ''}
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.gpu = gpu

        # se prepara el pipeline del modelo Llama3.2
        self._init_model()


    def _init_model(self):
        """
        Initializes the text generation model and tokenizer for Llama3.2.
        """

        # comprobar si hay GPU disponible
        use_gpu = self.gpu and torch.cuda.is_available()

        # cargar tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(__llama_path__)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # evitar warnings y mejorar compatibilidad con Llama-3
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Llama funciona mejor con padding a la izquierda en generación causal
        self.tokenizer.padding_side = "left"

        # cargar modelo
        model = AutoModelForCausalLM.from_pretrained(
        __llama_path__,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
        )

        # almacenar el modelo directamente (no crear pipeline innecesario)
        self.model = model

        logging.info(f"Modelo Llama3.2 cargado en {'GPU' if use_gpu else 'CPU'}.")


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
            self.messages = [self.prompt] + self.messages[-MAX_HISTORY:]

        # comprobar modelo
        if self.model is None:
            logging.error('Modelo no inicializado antes de generar texto')
            raise RuntimeError('Modelo no inicializado')

        # preparar tokens usando la plantilla del tokenizer para chat Llama-3
        input_encoding = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # determinar dispositivo robustamente a partir del modelo
        device = next(self.model.parameters()).device

        # mover tokens al dispositivo y preparar input_ids (apply_chat_template suele devolver Tensor)
        if isinstance(input_encoding, dict):
            input_ids = input_encoding["input_ids"].to(device)
        else:
            input_ids = input_encoding.to(device)

        # truncar prompt si excede la ventana del modelo
        max_pos = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.tokenizer, "model_max_length", None)
        if max_pos is not None:
            max_input_len = max(1, int(max_pos) - int(max_tokens))
            if input_ids.dim() == 1:
                input_seq_len = input_ids.shape[0]
            else:
                input_seq_len = input_ids.shape[-1]
            if input_seq_len > max_input_len:
                if input_ids.dim() == 1:
                    input_ids = input_ids[-max_input_len:]
                else:
                    input_ids = input_ids[:, -max_input_len:]

        # asegurar input_ids 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # attention_mask correcto: True donde no hay padding
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # generar con el modelo directamente (más robusto para modelos instruct)
        with torch.no_grad():
            outputs = self.model.generate(
                **generate_kwargs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # obtener solo los tokens generados (sin prompt) y decodificar
        generated_tokens = outputs[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

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