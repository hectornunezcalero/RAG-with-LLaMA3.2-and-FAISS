# Pasos para el funcionamiento de la IA (hasta ahora local).

## 1. Extracción de datos: scripts/data_extractor.py
Extrae texto de archivos PDF del directorio pdfdata y guarda el contenido en formato .txt en el directorio txtdata.
_python scripts/data_extractor.py_
<br>
## 2. Vectorización de datos: scripts/vectorizer.py
Divide los textos en chunks, genera vectores para cada chunk utilizando un modelo de embeddings y almacena los vectores en una base de datos FAISS.
_python scripts/vectorizer.py_
<br>
## 3. Ejecución del servidor: llama_server.py
Inicia un servidor Flask que utiliza la base de datos FAISS para realizar búsquedas de contexto y genera respuestas utilizando el modelo Llama.
_python llama_server.py_
<br>
## 4. Ejecución del cliente: scripts/llama_run.py
Proporciona una interfaz gráfica (GUI) para enviar preguntas al servidor y recibir respuestas generadas por el modelo.
_python scripts/llama_run.py_