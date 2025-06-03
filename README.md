# 🧠 LLaMA 3.2 RAG System – Local AI Search with Contextual Retrieval

Este repositorio contiene un sistema completo para la integración local de un modelo **LLaMA 3.2** (versión 3B) en un entorno **RAG (Retrieval-Augmented Generation)**. El objetivo es permitir la búsqueda y consulta sobre documentos locales mediante IA generativa y recuperación contextual.

---

## 📁 Estructura del Proyecto

```
📦 main_dir/
├── llama_client_local.py    # Cliente para consultas y sesiones (Tkinter + requests)
├── llama_server_local.py    # Servidor Flask que aloja LLaMA 3.2 y gestiona generación de texto
├── vectorizer.py            # Vectorización, embeddings y base de datos FAISS
├── data_extractor.py        # Extracción de texto desde PDF/TXT y preparación de corpus
├── pdfdata/                 # Carpeta con PDFs originales
├── txtdata/                 # Carpeta con texto plano (postprocesado)
├── vector_db/               # Base de datos vectorial generada con FAISS
├── requirements.txt         # Dependencias necesarias para el entorno virtual
├── setup_venv_windows.txt   # Instrucciones para configurar el entorno virtual en Windows
└── README2.md               # Archivo README del proyecto
```

---

## ⚙️ Requisitos previos

Antes de comenzar, asegúrate de cumplir con lo siguiente:

- Sistema operativo: **Windows 10/11** (con soporte para Python).
- **Python 3.10 o 3.11** instalado y añadido al PATH.
- Tener `pip` actualizado:  
  ```bash
  python -m pip install --upgrade pip
  ```
- Acceso local a **LLaMA 3.2** (previamente autorizado por Meta y descargado vía Hugging Face).
- Instalación de **Git** (opcional pero recomendado).
- Uso de una terminal como PowerShell, Terminal de VSCode o Git Bash.

---

## 🐍 Creación del entorno virtual (Windows)

Sigue estos pasos para configurar el entorno virtual en Windows:

```plaintext
# setup_venv_windows.txt

1. Abre PowerShell o Git Bash en la carpeta raíz del proyecto.

2. Ejecuta el siguiente comando para crear un entorno virtual:
   python -m venv venv

3. Activa el entorno virtual:
   .\venv\Scripts\activate

4. Instala todas las dependencias necesarias:
   pip install -r requirements.txt

```

Este proceso crea un entorno limpio y aislado, evitando conflictos con otras instalaciones de Python.

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

## 🚀 Ejecución del sistema

### 1. Extraer y vectorizar documentos

Primero, extrae el texto de los PDFs y guárdalo en formato `.txt`:

```bash
python data_extractor.py
```

Luego genera los embeddings y crea el índice FAISS:

```bash
python vectorizer.py
```

Esto gestionará `txtdata/` contruyendo el índice en `vector_db/`.

---

### 2. Levantar el servidor Flask con LLaMA

Abre una terminal nueva:

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

---

## 📥 ¿Cómo obtener acceso a LLaMA 3.2?

1. Accede a la página oficial del modelo en Hugging Face:  
   👉 [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)

2. Rellena el formulario de solicitud de Meta:
   - Usa un email institucional si es posible.
   - Describe tu propósito (por ejemplo, "TFG sobre búsqueda con IA usando RAG").
   - Acepta los términos de licencia.

3. Una vez aprobado, podrás descargarlo con `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-3B")
```

---

## 🧠 ¿Cómo funciona el sistema?

1. **Extracción**: El texto se extrae de PDFs y se limpia.
2. **Vectorización**: Se genera una base de vectores (embeddings) del texto con SentenceTransformers.
3. **Recuperación**: Ante una consulta, se buscan los vectores más cercanos en FAISS.
4. **Generación**: Se construye un prompt con el contexto recuperado y se genera una respuesta utilizando el modelo LLaMA 3.2.

---

## 🔐 Notas importantes

- Asegúrate de tener suficiente memoria (RAM/GPU) para LLaMA 3.2. Para CPU, puede ser más lento.
- Toda la información sensible (claves, rutas a modelos) debe mantenerse fuera del código fuente público.
- Este sistema es para **uso académico o personal**. El uso comercial requiere autorización explícita de Meta AI.

---

## 🤝 Instituciones involucradas

Este proyecto forma parte de un **Trabajo de Fin de Grado (TFG)** en Ingeniería Telemática – Universidad de Alcalá.  

---

## 📄 Licencia

Este proyecto está sujeto a las licencias de uso personal y académico del modelo LLaMA de Meta AI.  
No redistribuyas el modelo ni lo uses con fines comerciales sin autorización.

---

**Autor**: Héctor Núñez Calero 
**Año**: 2025/2026 s
**Contacto**: *[hector.nunez@edu.uah.es]*
