# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

import requests
import time
import hashlib
import logging
from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

# Si tienes una ruta específica para el archivo de claves API:
__keys_path__ = "./keys.txt"

LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000


class Llama3Server:
    def __init__(self, host='127.0.0.1', port=LLAMA_PORT):
        """
        Servidor Llama3.2 API en local.
        """
        logging.basicConfig(level=logging.INFO)
        logging.info('[+] Llama3.2-Server initialized (LOCAL).')

        self.app = Flask(__name__)

        # Carga el modelo GGUF (ajusta la ruta si hace falta)
        model_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

        self.llama = Llama(model_path=model_path, n_ctx=4096)

        # Registrar la ruta /request
        self.app.route('/request', methods=['POST'])(self.query)

        # Lanzar el servidor
        self.app.run(host=host, port=port)

    def query(self):
        key = request.headers.get('Authorization')
        session = request.headers.get('Session')
        data = request.json

        # Crear nueva sesión si es necesario
        if session == '0':
            session = 'si-' + hashlib.sha256(str(time.time()).encode()).hexdigest()[:20]

        # Validar API key
        is_valid, username = self.validate_api_key(key)
        if not is_valid:
            return jsonify({'response': 'Invalid Key', 'status_code': 401, 'query': data, 'session_id': 0})

        logging.info(f' [i] Procesando petición de {username} (sesión: {session}):\n{data}')

        try:
            task = data.get('task', 'generation')
            pooling = data.get('pooling', 'none')
            content = data.get('content', [])
            max_tokens = int(data.get('max_tokens', 2048))
            new_prompt = data.get('new_prompt')

            # Generar respuesta
            prompt = new_prompt if new_prompt else "\n".join(content)
            result = self.llama(prompt, max_tokens=max_tokens)

            answer = result["choices"][0]["text"].strip()

        except Exception as ex:
            logging.error(f"[!] Error: {ex}")
            return jsonify({
                'response': f'Runtime Error: {ex}',
                'status_code': 500,
                'query': data,
                'session_id': session
            })

        return jsonify({
            'response': answer,
            'status_code': 200,
            'query': data,
            'session_id': session
        })

    @staticmethod
    def validate_api_key(key):
        if not os.path.exists(__keys_path__):
            return False, "Archivo de claves no encontrado"

        with open(__keys_path__, 'r') as f:
            available_keys = f.read().splitlines()

        for entry in available_keys:
            if ':' in entry:
                stored_key, username = entry.split(':')
                if key == stored_key:
                    return True, username
        return False, ">Unknown<"


# Ejecutar el servidor en local
if __name__ == "__main__":
    Llama3Server()
