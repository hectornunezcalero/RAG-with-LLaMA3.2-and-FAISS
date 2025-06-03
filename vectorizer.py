# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación por Recuperación Aumentada (RAG)      #
#           con LLaMA 3.2 como asistente para consultas                 #
#           de artículos farmacéuticos del grupo de investigación       #
#           de la Universidad de Alcalá.                                #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: vectorizer.py                                           #
#       Funciones:                                                      #
#        1. Crear/Cargar y guardar la base de datos vectorial FAISS     #
#        2. Trocear texto con tokenizador de Hugging Face               #
#        3. Generar vectores con embeddings preentrenados (HF)          #
#        4. Detectar y añadir posibles archivos nuevos a la database    #
#        5. Regenerar la database si se detectan archivos eliminados    #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


from transformers import AutoTokenizer  # cargar el tokenizador del modelo de embeddings de Hugging Face
from langchain.schema import Document  # estructura estándar 'Document' para cada chunk: texto + metadatos
from langchain_community.vectorstores import FAISS  # instancia para base de datos vectorial FAISS destinado para las búsquedas por similitud
from langchain_community.docstore.in_memory import InMemoryDocstore  # almacén volatil en memoria RAM de esos objetos 'Document'
from langchain_huggingface import HuggingFaceEmbeddings  # sacar el modelo de embeddings de Hugging Face que convierte los chunks en vectores semánticos
import faiss  # crear y consultar la base de datos vectorial FAISS (versión CPU)
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import logging  # controlar y personalizar la salida de mensajes, avisos y errores

# se silencia el warnings que cree que no se va a chunkear y se va a exceder el límite de tokens
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# se carga el tokenizador del mismo modelo que usarás para embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")


# Dividir el texto en chunks para facilitar la búsqueda de similitudes vectoriales
def chunker(text, chunk_len, overlap):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    for i in range(0, len(input_ids), chunk_len - overlap):
        chunk_offsets = offsets[i:chunk_len + i]
        if not chunk_offsets:
            continue

        start = chunk_offsets[0][0]
        end = chunk_offsets[-1][1]
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# Comprobar la existe de la base de datos FAISS
def faiss_db_exists(output):
    faiss_index_path = os.path.join(output, "index.faiss")
    faiss_pkl_path = os.path.join(output, "index.pkl")
    return os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path)


# Cargar la base de datos FAISS existente desde disco
def load_faiss_db(output, embedding_model):
    faiss_db = FAISS.load_local(output, embedding_model, allow_dangerous_deserialization=True) # se carga index.faiss e index.pkl
    return faiss_db


# Crear o cargar la base de datos FAISS desde disco o inicializa una nueva
def create_faiss_db(embedding_model):
    # primeramente, el índice-vector, docstore y la relación índice-vector<->docstore se gestiona en la RAM :
    dimension = len(embedding_model.embed_query("test")) # dimensión vectorial para el modelo de embeddings (384 dimensiones para 'all-MiniLM-L12-v2')
    index = faiss.IndexFlatL2(dimension) # objeto FAISS con los índices y vectores N-dimensionales de la bbdd FAISS
    """ Estructura de index:
        0 : [0.19, 0.21, 0.36, ...]  # embedding del chunk 0
        1 : [0.41, 0.22, 0.57, ...]  # embedding del chunk 1
        2 : [0.97, 0.77, 0.14, ...]  # embedding del chunk 2
    """

    docstore = InMemoryDocstore({})  # almacén de estructuras de documentos sobre cada chunk, con data y metadata
    """ Estructura de docstore:
    { 
        '0': Document(
            page_content="primer chunk del archivo1.txt.",
            metadata={"source": "archivo1.txt", "chunk_index": 0}
        ),
        '1': Document(
            page_content="segundo chunk del archivo1.txt.",
            metadata={"source": "archivo1.txt", "chunk_index": 1}
        ),
        '2': Document(
            page_content="primer chunk del archivo2.txt.",
            metadata={"source": "archivo2.txt", "chunk_index": 0}
        ),
        ...
    }
    """

    index_to_docstore_id = {}  # diccionario a modo de mapa para relacionar los IDs de los índices de los vectores con los IDs de los documentos almacenados (chunk_data-chunk_metadata)
    """ Estructura de index_to_docstore_id:
    index_to_docstore_id = {
        0: '0',  # vector 0 -> docstore '0'
        1: '1',  # vector 1 -> docstore '1' 
        2: '2',  # vector 2 -> docstore '2' 
    }
    """

    faiss_db = FAISS(embedding_model, index, docstore, index_to_docstore_id)  # instancia de la base de datos FAISS con el modelo de embeddings, índice, almacén de documentos y relación índice<->docstore
    return faiss_db


# Vectorizar y añadir los nuevos archivos .txt encontrados en el directorio a la base FAISS
def vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output):
    existing_sources = {
        doc.metadata["source"]
        for doc in faiss_db.docstore._dict.values()
    }
    new_docs = []
    updated_files = 0

    for dirpath, _, files in os.walk(texts_dir):
        for archivo in files:
            if archivo.endswith(".txt"):
                full_path = os.path.join(dirpath, archivo)
                rel_path = os.path.relpath(full_path, texts_dir)  # esto será el "source" con subcarpeta incluida

                if rel_path in existing_sources:
                    continue

                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = chunker(content, chunk_len, overlap)

                for i, chunk in enumerate(chunks):
                    new_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": rel_path, "chunk_index": i}
                    ))

                updated_files += 1

        # una vez procesados todos los archivos, se añaden los documentos a la RAM
        print("Recopilando documentos nuevos en RAM...")
        faiss_db.add_documents(new_docs)

        # se vuelcan el índice y docstore en disco
        print("Guardando la nueva información en la base de datos...")
        faiss_db.save_local(output)

        files_num = len({doc.metadata["source"] for doc in faiss_db.docstore._dict.values()})
        print(f"\nActualizado(s) {updated_files} documento(s) nuevo(s) en la base de datos vectorial FAISS.")
        print(f"Total de archivos representados en la base de datos vectorial FAISS: {files_num}")


# Utilizar los argumentos clave de la función para crear/usar la base de datos FAISS y vectorizar nuevos archivos .txt
def save_processed_data(texts_dir, output, chunk_len, overlap):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

    current_files = {
        os.path.relpath(os.path.join(dirpath, f), texts_dir)
        for dirpath, _, files in os.walk(texts_dir)
        for f in files if f.endswith(".txt")
    }

    if not current_files:
        print(f"No se encontraron archivos .txt en el directorio '{texts_dir}' para vectorizar.")
        return

    if faiss_db_exists(output):
        # si la base de datos ya existe, solo la cargamos del disco
        print("Cargando base de datos...")
        faiss_db = FAISS.load_local(output, embedding_model, allow_dangerous_deserialization=True)

        # eliminamos los documentos y vectores indexados de los archivos que ya no existen por si se han descartado posteriormente
        stored_files = set(doc.metadata["source"] for doc in faiss_db.docstore._dict.values())
        deleted_files = stored_files - current_files
        added_files = current_files - stored_files

        if deleted_files:
            print(f"Se ha(n) detectado como eliminado(s) {len(deleted_files)} archivo(s). Actualizando por ello la base de datos...")
            faiss_db = create_faiss_db(embedding_model)
            vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output)
        elif added_files:
            print(f"Se ha(n) detectado {len(added_files)} archivo(s) nuevo(s). Procediendo a actualizar la base de datos...")
        else:
            print("La base de datos FAISS ya está actualizada.")
            return

    else:
        # si no existe, creamos una nueva base de datos FAISS vacía
        print("Creando nueva base de datos FAISS...")
        faiss_db = create_faiss_db(embedding_model)
        vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output)


# Función principal
if __name__ == "__main__":
    save_processed_data("./txtdata", "./vector_db", 180, 40)
