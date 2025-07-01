# 🧠 Retrieval-Augmented Generation (RAG) System using LLaMA 3.2 for Local Document Queries

This repository contains a complete system for the local integration of the LLaMA 3.2 model (version 3B) in a Retrieval-Augmented Generation (RAG) environment. Its purpose is to enable semantic search and querying over local documents using generative AI and contextual retrieval
---

## 📁 Project Structure (essential components)

```
📦 main_dir/
├── bin/
│   ├── data_extractor.py      # Extracts text from PDFs to TXT
│   ├── vectorizer.py          # Text chunking, embedding, FAISS vector DB
│   ├── server_run.py          # Launches the LLaMA-based Flask server
│   └── local_client.py        # GUI client to interact with the server
├── data/
│   ├── pdfdata/               # Original PDF documents
│   ├── txtdata/               # Extracted plain text files
│   └── vector_db/             # FAISS vector database
├── model/                     # Folder to store local models
├── src/
│   ├── __init__.py
│   ├── Llama32.py             # LLaMA 3.2 model logic
│   └── local_server.py        # Flask server with RAG logic
├── test/                      # Test scripts
├── venv/                      # Virtual environment
├── .gitignore
├── api_keys.txt               # needed API keys
├── README.md
├── requirements.txt           # Python dependencies
└── setup_venv_windows.txt     # Guide to set up the virtual environment on Windows

```

---

## ⚙️ Prerequisites

- Operating System: **Windows 10/11**.
- **Python 3.10 or 3.11** installed and added to PATH**.
- Latest pip version: 
  ```bash
  python -m pip install --upgrade pip
  ```
- Authorized access and local installation of **LLaMA 3.2**
- Use a terminal (preferably PowerShell)

---

## 📦 `requirements.txt`

Required Python libraries:

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
googletrans
```

---

## 🐍 Setting up the Virtual Environment (Windows)

```plaintext
# 1. Open PowerShell and navigate to the project directory:
cd main_dir

# 2. Create a virtual environment:
python -m venv venv

# 3. Activate the virtual environment:
.\venv\Scripts\activate

# 4. Install all required dependencies:
pip install -r requirements.txt
```

---

## 🚀  Running the System

### 1. Extract and vectorize documents

First of all, entry to binary code:

```bash
cd bin
```

Extract text from PDFs into plain .txt files:

```bash
python data_extractor.py
```

Then vectorize the extracted text into semantic embeddings and store them in a FAISS database:

```bash
python vectorizer.py
```

This will process the data/txtdata/ folder and generate embeddings in data/vector_db/.
---

### 2. Launch the Flask Server with LLaMA 3.2

Start the backend server that serves LLaMA 3.2 as the LLM:

```bash
python local_server.py
```

This will expose a local server at http://127.0.0.1:5666/.

---

### 3. Run the GUI Client

With the server running, launch the local GUI client:


```bash
python local_client.py
```

A graphical window will open where you can input questions. The system retrieves relevant context from the FAISS database and uses LLaMA 3.2 to generate responses.

---

## 🎁 System Overview (RAG Workflow)

1. **Extraction**: Raw text is extracted from PDFs.
2. **Vectorization**: Text is chunked and transformed into semantic vectors.
3. **Retrieval**: Closest vectors are retrieved using FAISS for any query.
4. **Generation**: A prompt is constructed with retrieved context, and LLaMA 3.2 generates a response.

---

## 🤝 Project Purpose

This project is part of a Final Degree Project (TFG) in Telematics Engineering at the University of Alcalá.

---

## 📄 License

This project follows Meta AI’s academic-use licensing for the LLaMA model.
Do not redistribute or use the model for commercial purposes without authorization.
---

**Author**: Héctor Núñez Calero.

**Academic Year:**: 2025/2026.

**Contact**: *[hector.nunez@edu.uah.es]*.
