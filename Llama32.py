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
#       Script: LLaMa32.py                                              #
#       Funciones principales:                                          #
#        1. Inicializar el modelo LLaMa 3.2 3B localmente               #
#        2. Configurar tareas de NLP: generación, extracción de         #
#           características, clasificación (no implementada)            #
#        3. Gestionar tokenizador y pipeline de transformers            #
#        4. Ejecutar la tarea seleccionada y manejar la generación      #
#           y procesamiento de textos                                   #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import numpy as np
import torch
from transformers import pipeline, AutoTokenizer
import logging

__llama_path__ = "./model"


# Main class:
class Llama3:
    def __init__(self, max_tokens: int = 1024, pooling: str = 'none', task: str = 'generation', gpu: bool = True):
        """
        Llama3 model for different tasks.
        :param max_tokens: The maximum number of tokens.
        :param pooling: The pooling method for the non-text-generation tasks.
        :param task: The task.
        :param gpu: Use GPU.
        """
        self._task_map = {
            'generation': ('text-generation', self.text_generation_task, True),
            'feature_extraction': ('feature-extraction', self.feature_extraction_task, True),
            'classification': ('text-classification', self.not_implemented_task, False),
            'zero_shot_classification': ('zero-shot-classification', self.not_implemented_task, False),
            'sentiment_analysis': ('sentiment-analysis', self.not_implemented_task, False)
        }
        available_pooling = ['none', 'mean', 'max', 'min', 'sum']
        pooling = pooling.lower()

        # Ops checking:
        if pooling not in available_pooling:
            raise ValueError(f'Pooling must be one of {available_pooling}')

        if not isinstance(max_tokens, int):
            raise ValueError(f'Tokens must be an integer.')

        if max_tokens < 1:
            raise ValueError(f'Tokens must be greater than 0.')

        # Initialize:
        self.pipe = None
        self.tokenizer = None
        self.task = None
        self.messages: list = list()
        self.prompt: dict = {}
        self.max_tokens: int = max_tokens
        self.pooling = pooling
        self.gpu = gpu

        # Set up the prompt:
        self.set_task(task)
        self.set_prompt('You are an useful assistant.')
        logging.info(f'[+] Llama3.2 model initialized with task {task}.')

    @property
    def tasks(self) -> list:
        """
        This function returns the task.
        :return: The task.
        """
        return list(self._task_map.keys())

    def set_task(self, task: str) -> None:
        """
        This function sets the task.
        :param task: The task.
        """
        task = task.lower()

        if task not in self._task_map:
            raise ValueError(f'Task must be one of {self._task_map.keys()}')

        self.task = task
        # Get tokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(__llama_path__)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Free the GPU memory for new pipe:
        del self.pipe
        self.pipe = pipeline(
            self._task_map[task][0],
            tokenizer=self.tokenizer,
            model = __llama_path__,
            torch_dtype = torch.bfloat16,
            device_map = 0 if self.gpu else 'cpu',
        )
        logging.info(f'[i] Task set to {task}.')

    def set_prompt(self, prompt: str):
        """
        This function sets the prompt and resets the messages.
        :param prompt: The prompt.
        """
        self.prompt = {'role': 'system', 'content': prompt}
        self.messages = [self.prompt]
        logging.info(f'[i] Prompt set to <{prompt}>')

    def text_generation_task(self, text: list[str], max_tokens: int = None):
        """
        This function generates text.
        :param text: The current text.
        :param max_tokens: The maximum number of tokens.
        :return: The generated text.
        """
        self.messages.append({'role': 'user', 'content': text})
        self.messages = self.pipe(self.messages, max_length = max_tokens)[0]['generated_text']
        return self.messages[-1]

    def feature_extraction_task(self, text: list[str], max_tokens: int = None):
        """
        This function extracts the features.
        :param text: The current text.
        :param max_tokens: The maximum number of tokens.
        :return: The extracted features.
        """
        embeddings = list()
        responses = self.pipe(text, max_length=max_tokens, padding=True)
        for response in responses:
            array_response = np.squeeze(np.array(response))
            # Apply pooling:
            if self.pooling == 'mean':
                embeddings.append(np.mean(array_response, axis = 0))
            elif self.pooling == 'max':
                embeddings.append(np.max(array_response, axis = 0))
            elif self.pooling == 'min':
                embeddings.append(np.min(array_response, axis = 0))
            elif self.pooling == 'sum':
                embeddings.append(np.sum(array_response, axis = 0))
            else:
                embeddings.append(array_response)
        return embeddings

    def not_implemented_task(self, text: list[str], max_tokens: int = None):
        """
        This function is called when the task is not implemented.
        :param text: The current text.
        :param max_tokens: The maximum number of tokens.
        :return: The output.
        """
        raise NotImplementedError(f'Task {self.task} is not implemented yet.')


    def __call__(self, *args, **kwargs):
        """
        This function is called when the object is called.
        :param args: An iterable with texts to be processed.
        :param kwargs: The keyword arguments.
        :return: The output of the model.
        """
        # Check if kwargs has 'tokens':
        max_tokens = self.max_tokens
        if 'tokens' in kwargs:
            # Compute the pipe:
            max_tokens = kwargs['tokens']
            # Check if the tokens are correct:
            if not isinstance(max_tokens, int):
                raise ValueError(f'Tokens must be an integer.')
            # Check if the tokens are in the correct range:
            if max_tokens < 1:
                raise ValueError(f'Tokens must be greater than 0.')

        # Compute the pipe for the given task:
        if self._task_map[self.task][2]:
            task_result = self._task_map[self.task][1](list(args), max_tokens = max_tokens)
        else:
            task_result = self._task_map[self.task][1](list(args))
        logging.info(f'[i] Task {self.task} executed with {len(args)} texts.')
        return task_result


    def __repr__(self):
        return f'<Llama3.2(task={self._task_map[self.task][0]}, max_tokens={self.max_tokens})>'
