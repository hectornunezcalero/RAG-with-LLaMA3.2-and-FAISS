## - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


from xmlrpc.client import ResponseError
import requests
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
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


class Llama3GUI:
    def __init__(self, llama_client):
        self.client = llama_client
        self.window = tk.Tk()
        self._prev_task = None
        self.window.title("Llama3.2 Client Interface")

        self.input_text, self.task_var, self.task_options, self.task_dropdown, self.send_button, self.output_text = \
            (None, None, None, None, None, None)

        # Setup layout
        self.setup_widgets()

    def setup_widgets(self):
        # Text entry for user input
        self.input_text = scrolledtext.ScrolledText(self.window, height=10, width=70)
        self.input_text.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        # Dropdown menu for selecting task
        self.task_var = tk.StringVar()
        self.task_options = ['generation', 'feature_extraction', 'classification', 'zero_shot_classification',
                             'sentiment_analysis']
        self.task_dropdown = ttk.Combobox(self.window, textvariable=self.task_var, values=self.task_options,
                                          state='readonly')
        self.task_dropdown.grid(row=1, column=0, padx=10, sticky='ew')
        self.task_dropdown.current(0)  # Default to generation

        # Button to send request
        self.send_button = tk.Button(self.window, text="Send Request", command=self.send_request)
        self.send_button.grid(row=1, column=1, padx=10, sticky='ew')

        # Text display for server response
        self.output_text = scrolledtext.ScrolledText(self.window, height=20, width=70)
        self.output_text.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

    def send_request(self):
        # Get the user input
        content = self.input_text.get('1.0', tk.END).strip()
        task = self.task_var.get()

        # Set up and send request
        if self._prev_task != task:
            self._prev_task = task
            self.client.set_up(max_tokens=200_000, task=task)

        try:
            server_response = self.client.process_request({"content": [content]})
            response_text = server_response.get("response", "No response received")
        except Exception as ex:
            response_text = f"Error: {ex}"

        # Display the response in the output text area
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, response_text)

    def run(self):
        self.window.mainloop()


# Assuming you have a working client instance named 'llama_client'
llama_client = Llama3CLI(api_key="<MASTERKEY>", server_ip="127.0.0.1")
app = Llama3GUI(llama_client)
app.run()