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
import numpy as np  # manejar datos de respuesta
from Llama32 import Llama3  # importar la clase Llama3 que dispone del LLM
import logging  # controlar y personalizar la salida de mensajes, avisos y errores
from flask import Flask, request, jsonify  # conseguir una API RESTful que permita la comunicación con el modelo

LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
SRV_IP = '192.168.XX.XX'

# se establece el archivo de texto con la clave de acceso a la API
__keys_path__ = "keys_path.txt"


# Clase para el servidor que maneja las peticiones y el LLM
class Llama3Server:
    def __init__(self, host = SRV_IP, port = LLAMA_PORT):
        # se despliega el servidor y el modelo Llama3.2
        logging.info('Servidor con Llama3.2 inicializado.')
        self.app = Flask(__name__)
        self.llama = Llama3()
        self.app.route('/request', methods=['POST'])(self.query)
        self.app.run(host, port)

    # Procesar la consulta recibida
    def query(self):
        # se recolecta de las cabeceras la clave de acceso y la sesión, además de obtenerse los datos de la consulta
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

        # se recolectan los parámetros de los datos recibidos
        data_content = data['content']
        data_max_tokens = data['max_tokens']
        new_prompt = data.get('new_prompt')

        # si se especifica un nuevo prompt, es porque se trata de una conversación nueva; si no, se está continuando una conversación
        if new_prompt:
            self.llama.set_prompt(new_prompt)

        # se deja al LLM que procese la consulta y genere una respuesta
        try:
            answer = self.llama(*data_content, tokens=data_max_tokens)
            if isinstance(answer, list):
                new_answer = list()
                for item in answer:
                    item = item.tolist() if isinstance(item, np.ndarray) else item
                    new_answer.append(item)
                answer = new_answer

        except Exception as ex:
            logging.error(f"Error: {ex}")
            return jsonify({'response': f'Runtime Error: {ex}', 'status_code': 500, 'query': data, 'session_id': session})

        # se prepara la respuesta a enviar al cliente
        response = {
            'response': answer,
            'status_code': 200,
            'query': data,
            'session_id': session
        }

        # finalmente, se devuelve la respuesta en formato JSON
        return jsonify(response)


    # Comprobar la validez de la clave de acceso enviada
    @staticmethod
    def validate_api_key(key):
        # se abre el archivo de claves y se lee su contenido:
        with open(__keys_path__, 'r') as f:
            available_keys = f.read().split('\n')

        # si la clave corresponde con alguna de las claves disponibles (clave:usuario), se devuelve True y el nombre de usuario
        for available_key_name in available_keys:
            available_key, username = available_key_name.split(':')
            if key == available_key:
                return True, username
        else:
            return False, ">Unknown<"