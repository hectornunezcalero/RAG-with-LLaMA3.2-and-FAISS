import time
import hashlib
import numpy as np
from Llama32 import Llama3
import logging
from flask import Flask, request, jsonify

LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000

__keys_path__ = "keys_path.txt"  # Path to the API keys file


# Server side:
class Llama3Server:
    def __init__(self, host='192.168.XX.XX', port=LLAMA_PORT):
        """
        This class is a server for the Llama3.2 API.
        :param host: The IP to host the service.
        :param port: The port to host the service.
        """
        logging.info('[+] Llama3.2-Server initialized.')
        self.app = Flask(__name__)
        # las sesiones se guardan en un diccionario:
        self.sessions = {}
        # Inline declaration:
        self.app.route('/request', methods=['POST'])(self.query)
        # se inicializa flask con el host y el puerto especificados, además de permitirse el uso de múltiples hilos:
        self.app.run(host, port, threaded=True)


    def query(self):
        """
        This function processes the query.
        :return: The response.
        """
        key = request.headers.get('Authorization')
        session = request.headers.get('Session')
        data = request.json

        # Create a new session or recover information:
        if session == '0':
            session = 'si-' + hashlib.sha256(str(time.time()).encode()).hexdigest()[:20]

        # Validate the key:
        is_valid, username = self.validate_api_key(key)
        if not is_valid:
            return jsonify({'response': 'Invalid Key', 'status_code': 401, 'query': data, 'session_id': 0})

        logging.info(f'[i] Processing request from {username} with session {session}:\n{data}')

        # Process the request:
        data_pooling = data['pooling']
        data_task = data['task']
        data_content = data['content']
        data_max_tokens = data['max_tokens']
        new_prompt = data.get('new_prompt', None)

        # Obtener o crear la instancia para cada sesión
        if session not in self.sessions:
            self.sessions[session] = Llama3()

        # establecer la tarea y el pooling de la instancia de Llama3:
        llama_instance = self.sessions[session]
        llama_instance.set_task(data_task)
        llama_instance.pooling = data_pooling

        # si se especifica un nuevo prompt, es porque se trata de una conversación nueva
        if new_prompt:
            self.llama.set_prompt(new_prompt)

        try:
            # procesar el contenido y obtener la respuesta de cada instancia de Llama3:
            answer = llama_instance(*data_content, tokens=data_max_tokens)
            if isinstance(answer, list):
                new_answer = list()
                for item in answer:
                    item = item.tolist() if isinstance(item, np.ndarray) else item
                    new_answer.append(item)
                answer = new_answer

        except Exception as ex:
            logging.error(f"[!] Error: {ex}")
            return jsonify({'response': f'Runtime Error: {ex}', 'status_code': 500,
                            'query': data, 'session_id': session})

        # Prepare the response:
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
        This function validates the API key.
        :param key: The key.
        :return: True if the key is valid, False otherwise, and the current user name.
        """
        with open(__keys_path__, 'r') as f:
            available_keys = f.read().split('\n')

        # Check if the key is in the list:
        for available_key_name in available_keys:
            available_key, username = available_key_name.split(':')
            if key == available_key:
                return True, username
        else:
            return False, ">Unknown<"