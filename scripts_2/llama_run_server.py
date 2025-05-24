# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import requests
import time
import hashlib
import numpy as np
from ..llama32 import Llama3
from ..__special__ import logging, __keys_path__
from flask import Flask, request, jsonify
LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000
"""
Llama3.2 API service.

QUERY:
{
    "content": ["TEXT1", "TEXT2", "TEXTn"]
    "new_prompt": "TEXT" or None
    "task": /"generation", "feature_extraction", "classification", "zero_shot_classification", "sentiment_analysis"/
    "pooling": /"none", "mean", "max", "min", "sum"/
    "max_tokens": int > 0.
}

RESPONSE:
{
    "response": <Depends on the task>
    "status_code": <Error message or 200 OK>
    "query": The given query.
    "session_id": <Session ID>
}
"""

# Client side:
class Llama3CLI:
    def __init__(self, api_key, server_ip):
        """
        This class is a client for the Llama3.2 API.
        :param api_key: The API key.
        :param server_ip: The server IP.
        """
        self.api_key = api_key
        self.base_ip = server_ip
        self.session_id = "0"
        self.setup_info = {}

    def process_request(self, data: dict):
        """
        This function processes the request.
        :param data: The data.
        {
            "content": ["TEXT1", "TEXT2", "TEXTn"]
            "new_prompt": "TEXT" or None
            "task": /"generation", "feature_extraction", "classification", "zero_shot_classification", "sentiment_analysis"/
            "pooling": /"none", "mean", "max", "min", "sum"/
            "max_tokens": int > 0.
        }
        :return: A dictionary with the response.
        {
            "response": <Depends on the task>
            "status_code": <Error message or 200 OK>
            "query": The given query.
            "session_id": <Session ID>
        }
        """
        headers = {'Authorization': self.api_key, 'Session': self.session_id}
        url = f"http://{self.base_ip}:{LLAMA_PORT}/request"

        if 'task' not in data:
            data['task'] = 'generation'
            logging.warning('[!] Task not specified. Defaulting to "generation".')
        if 'pooling' not in data:
            data['pooling'] = 'none'
            logging.warning('[!] Pooling not specified. Defaulting to "none".')
        if 'max_tokens' not in data:
            data['max_tokens'] = 1024
            logging.warning('[!] Max tokens not specified. Defaulting to 1024.')
        if 'content' not in data:
            return {"response": list(), "status_code": "No content", "query": data, "session_id": self.session_id}

        # Await response:
        try:
            response = requests.post(url, json=data, headers=headers)
        except Exception as ex:
            logging.error(f"[!] Connection error: {ex}")
            return {"response": list(), "status_code": "Host Unreachable", "query": data, "session_id": self.session_id}

        if response.status_code == 200:
            resp_json = response.json()
            self.session_id = resp_json['session_id']
            return resp_json
        else:
            logging.error(f"[!] Error {response.status_code}: {response.text}")
            return {"response": list(), "status_code": response.text, "query": data, "session_id": self.session_id}

    def set_up(self, max_tokens: int = 1024, pooling: str = 'none', task: str = 'generation'):
        """
        Llama3 model for different tasks.
        :param max_tokens: The maximum number of tokens.
        :param pooling: The pooling method for the non-text-generation tasks.
        :param task: The task.
        """
        self.setup_info = {
            'max_tokens': max_tokens,
            'pooling': pooling,
            'task': task,
            'new_prompt': None
        }

    def set_prompt(self, prompt: str):
        """
        This function sets the prompt.
        :param prompt: The prompt.
        """
        self.setup_info['new_prompt'] = prompt

    def __call__(self, *args, **kwargs):
        """
        This function processes the request.
        :param args: The content of the request.
        :param kwargs: Disregarded.
        :return:
        """
        data = self.setup_info
        data['content'] = list(args)
        response = self.process_request(data)['response']
        self.setup_info['new_prompt'] = None
        return response





    # Server side:
    class Llama3Server:
        def __init__(self, host='192.168.79.82', port=LLAMA_PORT):
            """
            This class is a server for the Llama3.2 API.
            :param host: The IP to host the service.
            :param port: The port to host the service.
            """
            logging.info('[+] Llama3.2-Server initialized.')
            self.app = Flask(__name__)
            self.llama = Llama3()
            # Inline declaration:
            self.app.route('/request', methods=['POST'])(self.query)
            # Run the server:
            self.app.run(host, port)

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

            # Set up llama:
            self.llama.set_task(data_task)
            self.llama.pooling = data_pooling
            if new_prompt:
                self.llama.set_prompt(new_prompt)
            # TODO: Session.

            try:
                # Process the content:
                answer = self.llama(*data_content, tokens=data_max_tokens)
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

    # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
    #                        END OF FILE                        #
    # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #