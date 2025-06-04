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
#       Script: data_extractor.py                                       #
#       Funciones principales                                           #
#        1. Manejar los directorios de pdfdata y txtdata.               #
#        2. Extraer texto de archivos PDF de pdfdata además de:         #
#           2.1 Estructurarlos por una o dos columnas.                  #
#           2.2 limpiarlos de espacios innecesarios.                    #
#        3. Guardar el texto extraído en archivos .txt en txtdata.      #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import fitz  # alias de PyMuPDF -> para extraer texto de archivos PDF
import os # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import re # limpiar y procesar texto mediante expresiones regulares


# Procesar los bloques de texto cada página del PDF, reduciendo espaciados innecesarios y existiendo solamente la separación de palabras y párrafos
def block_process(blocks):
    parts = []
    # evaluación de cada bloque de texto
    for block in blocks:
        content = ""
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    content += span["text"] + " "
            # se reemplaza en el contenido cualquier secuencia de uno o más espacios en blanco por un único espacio
            content = re.sub(r'\s+', ' ', content).strip()

        if content:
            parts.append(content)

    return "\n".join(parts)


# Extraer el texto de un PDF por bloques, teniendo en cuenta si cada hoja contiene una o dos columnas
def extract_txt(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"] # diccionario de información sobre los bloques de información: posición y líneas con spans de cada bloque de información
        """
            la variable 'blocks' contendrá un diccionario de información 
            de cada bloque de texto extraído ("blocks") con los siguientes campos:
        
             - bbox: Coordenadas del rectángulo delimitador del bloque de texto
                     (representa la posición y el tamaño del bloque en la página).
         
             - lines: Lista de líneas de texto dentro del bloque.
                      Cada línea es un diccionario que contiene:
         
                      - spans: Lista de fragmentos de texto de dentro de esa línea.
                               Cada fragmento incluye información como:
                                 * El propio texto del fragmento.
                                 * La fuente utilizada.
                                 * El tamaño del texto.
                                 * Otros atributos relacionados con el formato.
        """
        if not blocks:
            continue

        # para saber si la página tiene una o dos columnas, se trabajará con el ancho de la página
        page_wide = page.rect.width
        # se suman de las coordenadas 'x' de los bloques y se divide por el tamaño total para determinar la posición media de la posible "línea imaginaria" (si es que hay dos columnas no simétricas)
        average_x = sum(b["bbox"][0] for b in blocks) / len(blocks)

        # se separan los bloques en dos listas: izquierda y derecha, según su posición horizontal respecto la media
        left = [b for b in blocks if b["bbox"][0] < average_x]
        right = [b for b in blocks if b["bbox"][0] >= average_x]

        # se usará el porcentaje de bloques en el lado izquierdo para determinar si la página tiene una o dos columnas
        left_percent = len(left) / len(blocks)

        # se comprueba con ese porcentaje si la página tiene una o dos columnas
        if 0.3 < left_percent < 0.7:
            # mediante la función .sort() que ordena usando la key 'lambda', se toma cada bloque 'b' y devuelve su valor de 'bbox[1]', que representa la posición vertical 'y'
            # se ordena en dos columnas izq-der
            left.sort(key=lambda b: b["bbox"][1])
            right.sort(key=lambda b: b["bbox"][1])
            final_text += block_process(left)
            final_text += block_process(right)
        else:
            # se ordena en una sola columna
            blocks.sort(key=lambda b: b["bbox"][1])
            final_text += block_process(blocks)

    return final_text


# Extraer el texto de los PDFs y guardarlos como archivos de texto
def process_pdf(pdf_root, txt_root):
    extracted_count = 0
    deleted_count = 0

    # se recorre el directorio raíz que contiene los subdirectorios de PDFs
    for dirpath_pdfs, _, files in os.walk(pdf_root):
        pdf_files = [f for f in files if f.endswith(".pdf")]

        if not pdf_files:
            continue

        # se usará la misma subestructura en pdf_data que en txt_data
        rel_path = os.path.relpath(dirpath_pdfs, pdf_root) # ej: \Papers 20-21 (no se incluye el \pdf_data)
        txt_dir = os.path.join(txt_root, rel_path) # ej: .\txt_data\Papers 20-21 (usa rel_path para asemejar la subestructura que tiene pdf_data)

        # si no estaban creados los subdirectorios de texto, se crean
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # se eliminan los .txt cuyos pdfs ya no existen, si es que hay .txts
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith(".txt") and txt_file.replace(".txt", ".pdf") not in pdf_files:
                os.remove(os.path.join(txt_dir, txt_file))
                deleted_count += 1

        # Extraer PDFs nuevos
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath_pdfs, pdf_file) # \ej: pdf_data\Papers 20-21\articulo1.pdf
            txt_path = os.path.join(txt_dir, pdf_file.replace(".pdf", ".txt")) # \ej: .\txt_data\Papers 20-21\articulo1.txt

            # si el archivo de texto ya ha sido creado en ejecuciones anteriores (y no es un archivo vacío por error), se omite la extracción
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                continue

            # se llama al extractor de texto para ese PDF para su posterior escritura y almacenamiento en su versión .txt
            text = extract_txt(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Extracción de datos exitosa sobre \"{pdf_path}\"")
            extracted_count += 1

    if deleted_count > 0:
        print(f"Se han eliminado {deleted_count} textos correspondientes a PDFs inexistentes.")
    if extracted_count == 0:
        print("No hay PDFs por extraer, todos los archivos ya tienen su versión .txt.")
    else:
        print(f"Total de archivos extraídos a texto en esta ejecución: {extracted_count}")


# Función principal
if __name__ == "__main__":
    process_pdf("./pdfdata", "./txtdata")

