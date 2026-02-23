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
#       Script: local_server.py                                         #
#       Funciones principales:                                          #
#        1. Recibir la request del cliente con sus cabeceras y datos    #
#        2. Validad la clave API para autenticar y autorizar al cliente #
#        3. Enviar al LLM Llama3.2 la pregunta contextualizada para     #
#           que devuelva la respuesta a la pregunta, controlando si     #
#           es una nueva sesión o es una ya empezada                    #
#        4. Devolver la respuesta correspondiente al cliente            #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

import time  # usar 'time' para generar hashes sesiones únicas
import hashlib  # generar un hash único de la sesión
import os
import numpy as np  # manejar datos de respuesta
import logging  # controlar y personalizar la salida de mensajes, avisos y errores
from flask import Flask, request, jsonify  # conseguir una API RESTful que permita la comunicación con el modelo
from src.Llama32 import Llama3  # importar la clase Llama3 que dispone del LLM

LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SRV_IP = '127.0.0.1'

# se establece el archivo de texto con la clave de acceso a la API (ruta relativa a este fichero)
__keys_path__ = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'api_key.txt'))


class Llama3Server:
    """
    Flask server class to handle REST API requests for the Llama3.2 language model.
    The server accepts POST requests with queries, validates API keys,
    processes queries via the Llama3 model, and returns JSON responses.
    Attributes:
        app (Flask): Flask application instance.
        llama (Llama3): Instance of the Llama3 model to process queries.
    """
    def __init__(self, host = SRV_IP, port = LLAMA_PORT):
        """
        Initializes and runs the Flask server along with the Llama3 model.
        Args:
            host (str): IP address where the server will run.
            port (int): Port number on which the server listens.
        """
        logging.info('Servidor con Llama3.2 inicializado.')
        self.app = Flask(__name__)
        self.llama = Llama3()
        self.app.route('/request', methods=['POST'])(self.query)
        self.app.run(host, port)

    def query(self):
        """
        Handles POST requests to the '/request' endpoint.
        Extracts API key and session from headers, validates the key,
        processes the query using Llama3, and returns the result as JSON.
        Returns:
            Response JSON: Contains the generated response, HTTP status code,
                           original query data, and session ID.
        """
        key = request.headers.get('Authorization')
        session = request.headers.get('Session')
        data = request.json

        # si se trata de una nueva sesión, se genera un hash único para nombrarla
        if session == '0':
            session = 'si-' + hashlib.sha256(str(time.time()).encode()).hexdigest()[:20]

        # se valida la clave de acceso del usuario
        is_valid, username = self.validate_api_key(key)
        if not is_valid:
            return jsonify({'response': 'Invalid Key', 'status_code': 401, 'query': data, 'session_id': 0})

        logging.info(f'Procesando petición de {username} con número de sesión {session}:\n{data}')

        # se recolectan los parámetros de los datos recibidos (usar .get para evitar KeyError)
        data_content = data.get('content', '')
        data_max_tokens = data.get('max_tokens', getattr(self.llama, 'max_tokens', 1024))
        new_prompt = data.get('new_prompt')

        # si se especifica un nuevo prompt, es porque se trata de una conversación nueva; si no, se está continuando una conversación
        if new_prompt:
            self.llama.set_prompt(new_prompt)

        # se deja al LLM que procese la consulta y genere una respuesta
        try:
            print("Procesando consulta...")
            # si content es lista, expandir como argumentos; si no, pasar como único argumento
            if isinstance(data_content, list):
                answer = self.llama(*data_content, tokens=data_max_tokens)
            else:
                answer = self.llama(data_content, tokens=data_max_tokens)

            # normalizar numpy arrays dentro de la respuesta
            if isinstance(answer, list):
                print("Respuesta generada. Enviando al cliente...")
                new_answer = list()
                for item in answer:
                    item = item.tolist() if isinstance(item, np.ndarray) else item
                    new_answer.append(item)
                answer = new_answer

        except Exception as ex:
            logging.error(f"Error: {ex}")
            return jsonify({'response': f'Runtime Error: {ex}', 'status_code': 500, 'query': data, 'session_id': session}), 500

        # se prepara la respuesta a enviar al cliente
        response = {
            'response': answer,
            'status_code': 200,
            'query': data,
            'session_id': session
        }

        # finalmente, se devuelve la respuesta en formato JSON
        return jsonify(response)


    @staticmethod
    def validate_api_key(key):
        """
        Validates the provided API key against authorized keys stored in a file.
        Reads available keys from a file and checks if the provided key matches any.
        Args:
            key (str): API key provided in the request header.
        Returns:
            tuple: (bool, str)
                - True and username if the key is valid.
                - False and '>Unknown<' if the key is invalid.
        """
        # se abre el archivo de claves y se lee su contenido de forma segura
        try:
            with open(__keys_path__, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except FileNotFoundError:
            logging.error(f"API keys file not found: {__keys_path__}")
            return False, ">Unknown<"

        # si la clave corresponde con alguna de las claves disponibles (clave:usuario), se devuelve True y el nombre de usuario
        for available_key_name in lines:
            if ':' not in available_key_name:
                continue
            available_key, username = available_key_name.split(':', 1)
            if key == available_key:
                return True, username

        return False, ">Unknown<"

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#                               END OF FILE                             #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #