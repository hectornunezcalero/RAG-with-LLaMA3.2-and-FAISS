# 🧠 Sistema de Generación por Recuperación Aumentada (RAG) con LLaMA 3.2 como asistente para consultas

Este repositorio contiene un sistema completo para la integración local de un modelo **LLaMA 3.2** (versión 3B) en un entorno **RAG (Retrieval-Augmented Generation)**. El objetivo es permitir la búsqueda y consulta sobre documentos locales mediante IA generativa y recuperación contextual.

---

## 📁 Estructura del Proyecto (componentes esenciales)

```
📦 main_dir/
├── setup_venv_windows.txt   # Instrucciones para configurar el entorno virtual en Windows
├── requirements.txt         # Dependencias necesarias para el entorno virtual
├── pdfdata/                 # Carpeta con PDFs originales
├── txtdata/                 # Carpeta con texto plano (postprocesado)
├── data_extractor.py        # Extracción de texto desde PDF a TXT
├── vectorizer.py            # Vectorización, embeddings y base de datos FAISS
├── llama_server_local.py    # Servidor Flask que aloja LLaMA 3.2 y gestiona generación de texto
├── llama_client_local.py    # Cliente para consultas y sesiones (Tkinter + requests)
├── vector_db/               # Base de datos vectorial generada con FAISS
└── README.md                # Archivo README del proyecto
```

---

## ⚙️ Requisitos previos

Antes de comenzar, asegúrese de cumplir con lo siguiente:

- Sistema operativo: **Windows 10/11** (con soporte para Python).
- **Python 3.10 o 3.11** instalado y añadido al PATH.
- Tener `pip` actualizado:  
  ```bash
  python -m pip install --upgrade pip
  ```
- Acceso local a **LLaMA 3.2** (previamente autorizado por Meta y descargado vía Hugging Face).
- Uso de una terminal (preferiblemente PowerShell)

---

## 📦 `requirements.txt`

El archivo incluye dependencias esenciales como:

```txt
flask                        
faiss-cpu        
langchain           
langchain-community        
langchain-huggingface     
transformers                  
sentence-transformers         
torch                           
numpy                   
PyMuPDF                         
requests                        
accelerate                      
python-dotenv
```

---

## 🐍 Creación del entorno virtual (Windows)

Siga estos pasos para configurar el entorno virtual en Windows:

```plaintext
# setup_venv_windows.txt

1. Abra la terminal de PowerShell.

2. Navegue al directorio del proyecto:
   cd ruta\al\directorio\del\proyecto

3. Cree un entorno virtual con:
   python -m venv venv

4. Active el entorno virtual mediante:
   .\venv\Scripts\activate

5. Instale todas las dependencias necesarias usando:
   pip install -r requirements.txt

```

Este proceso construye un entorno limpio y aislado, evitando conflictos con otras instalaciones de Python.

---

## 🚀 Ejecución del sistema

### 1. Extraer y vectorizar documentos

En primer lugar, se extrae el texto de los archivos PDF del directorio pdfdata y guarda el contenido en formato .txt en el directorio txtdata.

```bash
python data_extractor.py
```

En segundo lugar, se divide los textos en chunks, genera vectores para cada chunk utilizando un modelo de embeddings y almacena los vectores en una base de datos FAISS.

```bash
python vectorizer.py
```

Esto gestionará `txtdata/` contruyendo los objetos 'document', vectores e índice en `vector_db/`.

---

### 2. Levantar el servidor Flask con LLaMA

Abra una terminal nueva:

```bash
python llama_server_local.py
```

Esto iniciará el backend en `http://127.0.0.1:5666/`, que espera consultas del cliente y responde con texto generado por el modelo LLaMA 3.2.

---

### 3. Ejecutar la interfaz gráfica del cliente

En otra terminal (manteniendo el servidor encendido):

```bash
python llama_client_local.py
```

Se abrirá una ventana gráfica que permite introducir preguntas. El sistema recuperará contexto relevante y generará respuestas usando el modelo.
Utilice la base de datos FAISS construida anteriormente para realizar recuperación de contexto a la hora de enviar el prompt con la query. 

---

## 📥 Cómo se obtiene el acceso a LLaMA 3.2

1. Acceda a la página oficial del modelo en Hugging Face:  
    [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)

2. Rellene el formulario de solicitud de Meta:
   - Use un email institucional si es posible.
   - Describa tu propósito (por ejemplo, "TFG sobre búsqueda con IA usando RAG").
   - Acepte los términos de licencia.

3. Una vez aprobado, se descargará con `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-3B")
```

---

## 🧠 Funcionamiento resumido del sistema RAG

1. **Extracción**: El texto se extrae de PDFs y se limpia.
2. **Vectorización**: Se genera una base de vectores (embeddings) del texto.
3. **Recuperación**: Ante una consulta, se buscan los vectores más cercanos en FAISS.
4. **Generación**: Se construye un prompt con el contexto recuperado y se genera una respuesta utilizando el modelo LLaMA 3.2.

---

## 🔐 Notas importantes

- Asegúrese de tener suficiente memoria (RAM/GPU) para LLaMA 3.2. Para CPU, puede ser más lento.
- Toda la información sensible (claves, rutas a modelos) debe mantenerse fuera del código fuente público.
- Este sistema es para **uso académico o personal**. El uso comercial requiere autorización explícita de Meta AI.

---

## 🤝 Justificación del proyecto

Este proyecto forma parte de un **Trabajo de Fin de Grado (TFG)** en Ingeniería Telemática – Universidad de Alcalá.  

---

## 📄 Licencia

Este proyecto está sujeto a las licencias de uso personal y académico del modelo LLaMA de Meta AI.  
No redistribuya el modelo ni lo uses con fines comerciales sin autorización.

---

**Autor**: Héctor Núñez Calero.

**Año**: 2025/2026.

**Contacto**: *[hector.nunez@edu.uah.es]*.
