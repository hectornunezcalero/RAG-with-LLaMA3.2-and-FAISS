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

# Procesador del POST en json
class Llama3CLI:
    def __init__(self, api_key, server_ip):
        self.api_key = api_key
        self.base_ip = server_ip
        self.session_id = "0"

    # Conexión y envío de query al servidor
    def process_request(self, data: dict):
        headers = {'Authorization': self.api_key, 'Session': self.session_id}
        url = f"http://{self.base_ip}:{LLAMA_PORT}/request"

        try:
            response = requests.post(url, json=data, headers=headers)
        except Exception as ex:
            logging.error(f"Connection error: {ex}")
            return {"response": "No se pudo conectar al servidor", "status_code": "Host Unreachable"}

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            return {"response": "Error en la respuesta del servidor", "status_code": response.status_code}

# Interfaz gráfica para la consulta del cliente
class Llama3GUI:
    def __init__(self, llama_client):
        self.client = llama_client
        self.window = tk.Tk()
        self.window.title("Llama3.2 - RAG Assistant")

        self.input_text = scrolledtext.ScrolledText(self.window, height=10, width=70)
        self.input_text.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        self.send_button = tk.Button(self.window, text="Send Request", command=self.send_request)
        self.send_button.grid(row=1, column=0, columnspan=2, pady=5, padx=10)

        self.output_text = scrolledtext.ScrolledText(self.window, height=20, width=70)
        self.output_text.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

    # Manejo de eventos para el botón de envío
    def send_request(self):
        content = self.input_text.get('1.0', tk.END).strip()
        if not content:
            self.output_text.insert(tk.END, "Ninguna pregunta añadida. Realice su pregunta por favor.\n")
            return

        response_data = self.client.process_request({"content": [content]})
        respuesta = response_data.get("response", "Respuesta vacía")

        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, respuesta)

    def run(self):
        self.window.mainloop()

# Ejecutar GUI
if __name__ == "__main__":
    llama_client = Llama3CLI(api_key="<MASTERKEY>", server_ip="127.0.0.1")
    app = Llama3GUI(llama_client)
    app.run()