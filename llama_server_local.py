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
#           de artículos farmacéuticos del grupo de investigación       #
#           de la Universidad de Alcalá                                 #
#                                                                       #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: llama_server_local.py                                         #
#       Funciones principales:                                          #
#        1. Cargar y prepararar el modelo LLaMA 3.2 desde HuggingFace   #
#        2. Inicializar el servidor Flask                               #
#        3. Definir el endpoint REST /request con autenticación         #
#        4. Procesar el prompt recibido y generar la respuesta          #
#        5. Gestión de errores, acceso y respuesta en formato JSON      #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import time  # medir tiempos o controlar pausas en la ejecución
import hashlib  # generar hashes, usándolos como identificadores o para comprobar la integridad
import torch  # backend PyTorch usado por cliente y vectorizer para ejecutar modelos de embeddings (no se importa explícitamente)
from flask import Flask, request, jsonify  # proporcionar la API REST al cliente
from dotenv import load_dotenv  # cargar variables de entorno desde un archivo .env, facilitando la configuración del entorno de ejecución
from transformers import AutoTokenizer, AutoModelForCausalLM  # cargar el tokenizador del modelo de embeddings

# se cargan las variables del archivo .env
load_dotenv()

API_KEY = None
HF_TOKEN = os.getenv("HF_TOKEN")  # Token de Hugging Face para el modelo privado LLaMa 3.2-3B

# ruta del archivo con las claves API
__keys_path__ = os.getenv("KEYS_PATH", "keys_path.txt")

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

# se cargan todas las claves API una vez
try:
    with open(__keys_path__, "r") as f:
        available_keys = [line.strip() for line in f if line.strip() and ':' in line]
except Exception as e:
    print(f"Error cargando archivo de claves API en {__keys_path__}: {e}")
    available_keys = []

@app.route("/")
def home():
    return "Servidor API LLaMA 3.2 3B con backend de Hugging Face"

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

    prompt = data.get("new_prompt")
    max_tokens = int(data.get("max_tokens", 4096))

    if not prompt:
        return jsonify({"error": "Falta 'new_prompt' en el cuerpo"}), 400


    is_valid, username = validate_api_key(auth)
    if not is_valid:
        return jsonify({'response': 'Acceso denegado: clave API inválida', 'status_code': 401, 'query': data, 'session_id': 0})

    print(f"Procesando petición del {username} con sesión {session}:\n{data}")


    try:
        # se tokeniza el prompt  y se adapta a los 'tensores' del modelo
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # se desactiva el cálculo de gradientes (aprendizaje) para ahorrar memoria y acelerar la inferencia
        with torch.no_grad():
            outputs = model.generate( # para generar el texto en base a los tensores de entrada y los parámetros proporcionados
                **inputs, # se pasan los tensores de entrada
                max_new_tokens=max_tokens,
                do_sample=True, # se activa el muestreo aleatorio para generar texto más variado
                temperature=0.7, # se ajusta el valor de creatividad (valores altos -> creatividad, valores bajos -> precisión)
                top_p=0.9 # se filtran los tokens según su probabilidad acumulada, mejorando la coherencia de la respuesta
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as ex:
        print(f"Error en ejecución: {ex}")
        return jsonify({'response': f'Error de ejecución: {ex}', 'status_code': 500,
                        'query': data, 'session_id': session})

    return jsonify({
        "response": response_text,
        "status_code": 200,
        "query": data,
        "session_id": session
    })


def validate_api_key(key):
    for line in available_keys:
        stored_key, username = line.split(':', 1)
        if key == stored_key:
            return True, username

    return False, ">Unknown<"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=LLAMA_PORT)








