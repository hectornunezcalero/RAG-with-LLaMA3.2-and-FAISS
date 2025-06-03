# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Proyecto de Fin de Grado:                                       #
#           Servidor REST API para LLaMA 3.2                            #
#           Implementación de backend para procesamiento de             #
#           consultas mediante modelo LLaMA con Flask                   #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: llama3_server.py                                        #
#       Funciones principales:                                          #
#          - Inicializar y arrancar el servidor Flask local             #
#          - Establecer endpoint REST /request para procesar consultas  #
#          - Validar las API Keys desde archivo                         #
#          - Gestionar las sesiones y respuesta JSON para cliente       #
#          - Manejar errores y logging                                  #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import time  # para medir tiempos o controlar pausas en la ejecución
import hashlib  # para generar hashes, usándolos como identificadores o para comprobar la integridad
import numpy as np  # para manejo numérico que conlleve arrays
import os  # para manejar rutas, directorios, archivos y operaciones del sistema de ficheros
from ..llama32 import Llama3  # se importa la clase Llama3 del módulo llama32
import logging  # para controlar y personalizar la salida de mensajes, avisos y errores
from flask import Flask, request, jsonify  # para proporcionar la API REST al cliente

# Definimos aquí la ruta al archivo keys_path.txt (mismo directorio que este script)
__keys_path__ = os.path.join(os.path.dirname(__file__), 'keys_path.txt')

# Calcula un puerto personalizado a partir del nombre 'llama3.2'
LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000


# Server side:
class Llama3Server:
    def __init__(self, host='127.0.0.1', port=LLAMA_PORT):
        """
        Este servidor ejecuta la API de Llama3.2 en entorno local.
        :param host: Dirección IP en la que se lanza el servidor (localhost para entorno local).
        :param port: Puerto en el que se lanza el servidor.
        """
        logging.info('[+] Llama3.2-Server inicializado en entorno local.')
        self.app = Flask(__name__)
        self.llama = Llama3()

        # Declara la ruta inline
        self.app.route('/request', methods=['POST'])(self.query)

        # Lanza el servidor Flask en local
        self.app.run(host, port)

    def query(self):
        """
        Esta función procesa la consulta que llega al endpoint /request.
        :return: La respuesta del modelo.
        """
        key = request.headers.get('Authorization')
        session = request.headers.get('Session')
        data = request.json

        # Si es una nueva sesión, genera un nuevo ID de sesión:
        if session == '0':
            session = 'si-' + hashlib.sha256(str(time.time()).encode()).hexdigest()[:20]

        # Valida la API Key:
        is_valid, username = self.validate_api_key(key)
        if not is_valid:
            return jsonify({'response': 'Invalid Key', 'status_code': 401, 'query': data, 'session_id': 0})

        logging.info(f'[i] Procesando petición de {username} con sesión {session}:\n{data}')

        # Extrae y configura los parámetros:
        data_pooling = data['pooling']
        data_task = data['task']
        data_content = data['content']
        data_max_tokens = data['max_tokens']
        new_prompt = data.get('new_prompt', None)

        self.llama.set_task(data_task)
        self.llama.pooling = data_pooling
        if new_prompt:
            self.llama.set_prompt(new_prompt)

        try:
            # Procesa la entrada con el modelo
            answer = self.llama(*data_content, tokens=data_max_tokens)
            if isinstance(answer, list):
                new_answer = []
                for item in answer:
                    item = item.tolist() if isinstance(item, np.ndarray) else item
                    new_answer.append(item)
                answer = new_answer

        except Exception as ex:
            logging.error(f"[!] Error: {ex}")
            return jsonify({
                'response': f'Runtime Error: {ex}',
                'status_code': 500,
                'query': data,
                'session_id': session
            })

        # Devuelve la respuesta al cliente:
        response = {
            'response': answer,
            'status_code': 200,
            'query': data,
            'session_id': session
        }

        return jsonify(response)

    @staticmethod
    def validate_api_key(key):
        """
        Valida la API key contra las disponibles en el archivo.
        :param key: Clave de autorización enviada por el cliente.
        :return: (boolean, nombre de usuario si es válido)
        """
        try:
            with open(__keys_path__, 'r') as f:
                available_keys = f.read().split('\n')
        except Exception as e:
            logging.error(f"Error leyendo el archivo de claves: {e}")
            return False, ">Unknown<"

        for available_key_name in available_keys:
            if ':' not in available_key_name:
                continue
            available_key, username = available_key_name.split(':')
            if key == available_key:
                return True, username

        return False, ">Unknown<"


# Solo ejecuta si se lanza como script principal
if __name__ == '__main__':
    Llama3Server()




