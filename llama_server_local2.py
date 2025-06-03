# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Proyecto de Fin de Grado:                                       #
#           Sistema de Generación por Recuperación Aumentada (RAG)      #
#           con LLaMA 3.2 como asistente para consultas                 #
#           de artículos farmacéuticos.                                 #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: llama_server.py                                         #
#       Funciones principales:                                          #
#        1. Cargar y prepararar el modelo LLaMA 3.2 desde HuggingFace   #
#        2. Inicializar el servidor Flask                               #
#        3. Definir el endpoint REST /request con autenticación         #
#        4. Procesar el prompt recibido y generar la respuesta          #
#        5. Gestión de errores, acceso y respuesta en formato JSON      #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import os
import time
import hashlib
import torch
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Carga variables del archivo .env
load_dotenv()

# Variables de entorno
API_KEY = None  # Ya no fija, sino se valida con archivo
HF_TOKEN = os.getenv("HF_TOKEN")  # Token Hugging Face para modelos privados

# Ruta archivo con claves API (ajusta si quieres)
__keys_path__ = os.getenv("KEYS_PATH", "keys_path.txt")

# Puerto personalizado (como antes)
LLAMA_PORT = sum([ord(c) for c in 'llama3.2']) + 5000

# Inicializa Flask
app = Flask(__name__)

print("Cargando modelo LLaMA 3.2-3B con Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    use_auth_token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",  # Ajusta para GPU si hay
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)

model.eval()
print("Modelo cargado correctamente.")

# Carga todas las claves API una vez
try:
    with open(__keys_path__, "r") as f:
        available_keys = [line.strip() for line in f if line.strip() and ':' in line]
except Exception as e:
    print(f"[!] Error cargando archivo de claves API en {__keys_path__}: {e}")
    available_keys = []

@app.route("/")
def home():
    return "Servidor API LLaMA 3.2 con backend Hugging Face"

@app.route("/request", methods=["POST"])
def llama_request():
    if request.method != "POST":
        return jsonify({"error": "Método no permitido"}), 405

    auth = request.headers.get("Authorization")
    if not auth:
        return jsonify({"error": "Falta header Authorization"}), 401

    session = request.headers.get("Session", "0")
    if session == "0":
        session = "si-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:20]

    data = request.get_json()
    if not data:
        return jsonify({"error": "Cuerpo JSON vacío"}), 400

    # Usa el campo 'new_prompt' para generar texto (igual que antes)
    prompt = data.get("new_prompt")
    max_tokens = int(data.get("max_tokens", 256))

    if not prompt:
        return jsonify({"error": "Falta 'new_prompt' en el cuerpo"}), 400

    # Validación de clave API usando el archivo cargado
    is_valid, username = validate_api_key(auth)
    if not is_valid:
        return jsonify({'response': 'Acceso denegado: clave API inválida', 'status_code': 401, 'query': data, 'session_id': 0})

    # Logging
    print(f"[i] Procesando petición de {username} con sesión {session}:\n{data}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as ex:
        print(f"[!] Error en ejecución: {ex}")
        return jsonify({'response': f'Error de ejecución: {ex}', 'status_code': 500,
                        'query': data, 'session_id': session})

    return jsonify({
        "response": response_text,
        "status_code": 200,
        "query": data,
        "session_id": session
    })


def validate_api_key(key):
    """
    Valida la clave API usando la lista cargada disponible_keys.
    """
    for line in available_keys:
        stored_key, username = line.split(':', 1)
        if key == stored_key:
            return True, username

    return False, ">Unknown<"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=LLAMA_PORT)








