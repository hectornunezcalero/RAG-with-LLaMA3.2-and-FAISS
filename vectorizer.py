# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática   -   Curso 2025/2026            #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación Aumentada por Recuperación (RAG)      #
#           con LLaMA 3.2 como asistente para consultas                 #
#           sobre artículos farmacéuticos que dispone el                #
#           grupo de investigación de la Universidad de Alcalá.         #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: vectorizer.py                                           #
#       Funciones principales:                                          #
#        1. Crear/Cargar la base de datos vectorial FAISS               #
#        2. Detectar cambios de archivos que impliquen a la bbdd        #
#        3. Chunkear los nuevos textos para las relaciones semánticas   #
#        4. Generar embeddings y almacenarlos en la base de datos       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


from transformers import AutoTokenizer  # cargar el tokenizador del modelo de embeddings de Hugging Face
from langchain_community.vectorstores import FAISS  # instancia para base de datos vectorial FAISS destinada a las búsquedas por similitud
from langchain.schema import Document  # estructura estándar 'Document' para cada chunk: texto + metadatos
from langchain_community.docstore.in_memory import InMemoryDocstore  # almacén volatil en memoria RAM de esos objetos 'Document'
from langchain_huggingface import HuggingFaceEmbeddings  # usar el modelo de embeddings de Hugging Face que convierte los chunks en vectores semánticos
import faiss  # consultar la base de datos vectorial FAISS (versión CPU)
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import logging  # controlar y personalizar la salida de mensajes, avisos y errores

TXT_ROOT_PATH = ".\\txtdata"
DATABASE_PATH = ".\\vector_db"
CHUNK_LEN = 180
OVERLAP = 30


# se silencia el warning que cree que no se va a chunkear y se va a exceder el límite de tokens
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# se carga el modelo tokenizador y vectorizador
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')


# Dividir el texto en chunks para facilitar la búsqueda de similitudes semánticas
def chunker(text, chunk_len, overlap):

    # 'encoding' recopilará los IDs de los tokens del modelo y los offsets de cada token en el texto para su posterior uso
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    tokens_ids = encoding["input_ids"]
    """ Estructura de tokens_ids:
        "La Universidad de Alcalá posee..." : [101, 1045, 2572, 1037, 2143, ...]
    """

    tokens_offsets = encoding["offset_mapping"]
    """ Estructura de tokens_offsets:
        "La Universidad de Alcalá posee..." : [(0, 1), (2, 12), (13, 14), (15, 20), (21, 25), ...]
    """

    # del texto original, los chunks serán 180 tokens seguidos donde los 40 primeros son el solapamiento
    # con el chunk anterior para tener contexto y los 140 siguientes es el resto de información
    chunks = []
    for i in range(0, len(tokens_ids), chunk_len - overlap):
        # se almacenan los offsets de los tokens del chunk actual
        chunk_offsets = tokens_offsets[i:chunk_len + i]
        if not chunk_offsets:
            continue

        # se entiende como 'start' al primer elemento del primer offset del chunk (comienzo del primer token)
        start = chunk_offsets[0][0]
        # se establece como 'end' al último elemento del último offset del token del chunk
        end = chunk_offsets[-1][1]
        # se forma el texto del chunk
        chunk_text = text[start:end].strip()

        if chunk_text and len(chunk_text) > 0:
            chunks.append(chunk_text)

    return chunks


# Comprobar la existencia de la base de datos FAISS
def faiss_db_exists(output):
    faiss_index_path = os.path.join(output, "index.faiss")
    faiss_pkl_path = os.path.join(output, "index.pkl")
    return (
            os.path.exists(faiss_index_path) and os.path.getsize(faiss_index_path) > 0 and
            os.path.exists(faiss_pkl_path) and os.path.getsize(faiss_pkl_path) > 0
    )


# Cargar la base de datos FAISS existente desde disco en memoria RAM
def load_faiss_db(output):
    faiss_db = FAISS.load_local(output, embedding_model, allow_dangerous_deserialization=True) # se carga index.faiss e index.pkl
    return faiss_db


# Crear la base de datos FAISS en memoria RAM
def create_faiss_db():

    # primeramente, se crean el índice-vector, docstore y la relación índice-vector<->docstore
    dimension = len(embedding_model.embed_query("test")) # dimensión vectorial para el modelo de embeddings (384 dimensiones para 'all-MiniLM-L12-v2')
    index = faiss.IndexFlatL2(dimension) # objeto faiss con los índices y vectores N-dimensionales de la bbdd FAISS
    """ Estructura de index:
    0 : [0.19, 0.21, 0.36, ...]  # embedding del chunk 0
    1 : [0.41, 0.22, 0.57, ...]  # embedding del chunk 1
    2 : [0.97, 0.77, 0.14, ...]  # embedding del chunk 2
    """

    docstore = InMemoryDocstore({})  # almacén de estructuras de documentos sobre cada chunk con índice, data y metadata
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

    index_to_docstore_id = {}  # diccionario a modo de mapa para relacionar los IDs de los índices de los vectores con los IDs de los documentos almacenados
    """ Estructura de index_to_docstore_id:
    {
        0: '0',  # vector 0 -> docstore '0'
        1: '1',  # vector 1 -> docstore '1' 
        2: '2',  # vector 2 -> docstore '2' 
    }
    """

    # se crea la instancia de la base de datos FAISS con el modelo de embeddings y los 3 componentes anteriores
    faiss_db = FAISS(embedding_model, index, docstore, index_to_docstore_id)  # instancia de la base de datos FAISS con el modelo de embeddings, índice, almacén de documentos y relación índice<->docstore
    return faiss_db


# Vectorizar y añadir los nuevos archivos .txt a la base de datos
def vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output):

    # se tendrán en cuenta los documentos de los chunks ya existentes en la base de datos para evitar duplicados
    print("Detectando los archivos de texto existentes en la base de datos...")
    existing_sources = {
        doc.metadata["source"]
        for doc in faiss_db.docstore._dict.values()
    }
    new_docs = []
    updated_files = 0

    # comprobación de la existencia de los documentos de los chunks
    print(f"Construyendo los contenidos a añadir desde '{texts_dir}'...")
    for dirpath, _, files in os.walk(texts_dir):
        for archivo in files:
            if archivo.endswith(".txt"):
                full_path = os.path.join(dirpath, archivo)
                rel_path = os.path.relpath(full_path, texts_dir)

                # si existe el archivo en la base de datos, se omite
                if rel_path in existing_sources:
                    continue

                #si no existe, se vectoriza el archivo
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # se recurre al chunkeo del contenido del archivo
                chunks = chunker(content, chunk_len, overlap)

                # por cada chunk, se crea un nuevo objeto Document con su contenido y metadatos
                for i, chunk in enumerate(chunks):
                    new_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": rel_path, "chunk_index": i}
                    ))

                updated_files += 1

    # una vez procesados todos los archivos, se añaden los documentos a la RAM
    print("Recopilando todos los nuevos documentos creados en la RAM...")
    faiss_db.add_documents(new_docs)

    # se vuelcan el índice y docstore en disco
    print("Guardando la nueva información en la base de datos FAISS local...")
    faiss_db.save_local(output)

    files_num = len({doc.metadata["source"] for doc in faiss_db.docstore._dict.values()})
    print(f"\nActualizado(s) {updated_files} archivo(s) nuevo(s) en la base de datos vectorial FAISS.")
    print(f"Sumando un total de {files_num} representados.")


# Crear/Actualizar en disco la base de datos FAISS con los archivos .txt procesados
def save_processed_data(texts_dir: str, output: str, chunk_len, overlap):

    # se tratará con los archivos .txt existentes en el directorio txt_data
    print("Detectando todos los archivos extraidos en formato texto...")
    current_files = {
        os.path.relpath(os.path.join(dirpath, f), texts_dir)
        for dirpath, _, files in os.walk(texts_dir)
        for f in files if f.endswith(".txt")
    }

    if not current_files:
        print(f"No se encontraron archivos .txt en el directorio '{texts_dir}' para vectorizar.")
        return

    # si la base de datos ya existe, solo se carga del disco
    print("Comprobando la existencia de la base de datos FAISS...")
    if faiss_db_exists(output):
        print(f"Cargando base de datos ubicada en '{output}'...")
        faiss_db = FAISS.load_local(output, embedding_model, allow_dangerous_deserialization=True)

        # se detectan archivos .txts que ya no existen por si fueron eliminados
        stored_files = set(doc.metadata["source"] for doc in faiss_db.docstore._dict.values())
        deleted_files = stored_files - current_files
        # además, se comprueba si hay archivos nuevos que no están en la base de datos
        added_files = current_files - stored_files

        # si se han eliminado archivos, se crea la base de datos de nuevo
        if deleted_files:
            print(f"Se ha(n) detectado como eliminado(s) {len(deleted_files)} archivo(s). Actualizando por ello la base de datos...")
            faiss_db = create_faiss_db()
            vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output)

        # si se han añadido archivos nuevos, se vectorizan y añaden a la base de datos
        elif added_files:
            print(f"Se ha(n) detectado {len(added_files)} archivo(s) nuevo(s). Procediendo a actualizar la base de datos...")
            vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output)

        # si no se han eliminado ni añadido archivos, la base de datos ya está actualizada
        else:
            print("La base de datos FAISS ya está actualizada.")
            return

    # si no existe, creamos una nueva base de datos vacía
    else:
        print("Creando nueva base de datos FAISS...")
        faiss_db = create_faiss_db()
        vectorize_new_txt_files(texts_dir, faiss_db, chunk_len, overlap, output)


# Función principal
if __name__ == "__main__":
    save_processed_data(TXT_ROOT_PATH, DATABASE_PATH, CHUNK_LEN, OVERLAP)
