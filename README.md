# Pasos para el funcionamiento del sistema RAG con Llama3.2 como LLM.

## 1. Extracción de datos: scripts/data_extractor.py
Extrae texto de archivos PDF del directorio pdfdata y guarda el contenido en formato .txt en el directorio txtdata.
<br>
_python scripts/data_extractor.py_
<br>
## 2. Vectorización de datos: scripts/vectorizer.py
Divide los textos en chunks, genera vectores para cada chunk utilizando un modelo de embeddings y almacena los vectores en una base de datos FAISS.
<br>
_python scripts/vectorizer.py_
<br>
## 3. Ejecución del servidor: llama_server.py (en constante funcionamiento) 
Se tiene en cuenta el servidor Flask que genera respuestas utilizando el modelo Llama3.2 3B ubicado en la EPS de la Universidad de Alcalá.
<br>
_python llama_server.py_
<br>
## 4. Ejecución del cliente: llama_client.py
Proporciona una interfaz gráfica (GUI) para enviar preguntas y por tanto el prompt al servidor, recibiendo respuestas generadas por el modelo.
Utiliza la base de datos FAISS para realizar recuperación de contexto a la hora de enviar el prompt con la query. 
<br>
_python llama_client.py_