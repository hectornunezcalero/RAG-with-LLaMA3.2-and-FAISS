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
#           de artículos farmacéuticos del grupo de investigación
#           de la Universidad de Alcalá.                                 #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Cotutor: Alberto Palomo Alonso                                  #
#       Tutor: Jorge Pérez Aracil                                       #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: data_extractor.py                                       #
#       Funciones:                                                      #
#        1. Manejar los subdirectorios de pdfdata/ y txtdata/           #
#        2. Extraer el texto de archivos PDF y limpiarlos               #
#        3. Detectar estructuras en columnas y ordena el contenido      #
#        4. Guardar el texto extraído en archivos .txt equivalentes     #
#        5. Eliminar archivos .txt huérfanos si su PDF ya no existe     #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import fitz  # alias de PyMuPDF -> para extraer texto de archivos PDF
import os # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import re # limpiar y procesar texto mediante expresiones regulares


# Procesar los bloques de texto cada página del PDF, reduciendo espaciados innecesarios y soloexisitiendo los de separación de palabras y párrafos
def block_process(blocks):
    parts = []
    for block in blocks:
        content = ""
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    content += span["text"] + " "
            # se reemplaza cualquier secuencia de uno o más espacios en blanco por un único espacio
            content = re.sub(r'\s+', ' ', content).strip()

        if content:
            parts.append(content)

    return "\n".join(parts)


# Extraer el texto de un PDF por bloques, teniendo en cuenta si cada hoja contiene una o dos columnas
def extract_txt(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"] # diccionario de información sobre la posición y contenido de cada bloque
        if not blocks:
            continue

        page_wide = page.rect.width
        # se suman de las coordenadas 'x' de los bloques para determinar la posición media de la posible "línea imaginaria" (por si hay dos columnas no simetricas)
        average_x = sum(b["bbox"][0] for b in blocks) / len(blocks) if blocks else page_wide / 2

        left = [b for b in blocks if b["bbox"][0] < average_x]
        right = [b for b in blocks if b["bbox"][0] >= average_x]

        total = len(blocks)
        left_percent = len(left) / total if total > 0 else 1

        # se define si la página tiene una o dos columnas, una vez comprobada el tamaño del lado izquierdo
        if 0.3 < left_percent < 0.7:
            # mediante la función .sort() que ordena usando la key 'lambda', se toma cada bloque 'b' y devuelve su valor de 'bbox[1]', que representa la posición vertical
            left.sort(key=lambda b: b["bbox"][1])
            right.sort(key=lambda b: b["bbox"][1])
            final_text += block_process(left)
            final_text += block_process(right)
        else:
            blocks.sort(key=lambda b: b["bbox"][1])
            final_text += block_process(blocks)

    return final_text.strip()


# Extraer el texto de los PDFs y guardarlos como archivos de texto
def process_pdf(pdf_root, txt_root):
    extracted_count = 0
    deleted_count = 0

    # se recorre el directorio raíz que contiene los directorios de PDFs
    for dirpath_pdfs, _, files in os.walk(pdf_root):
        pdf_files = [f for f in files if f.endswith(".pdf")]

        if not pdf_files:
            continue

        # se crea la misma subestructura en txt_root
        rel_path = os.path.relpath(dirpath_pdfs, pdf_root)
        txt_dir = os.path.join(txt_root, rel_path)

        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # se eliminan los .txt cuyos pdfs ya no existen
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith(".txt") and txt_file.replace(".txt", ".pdf") not in pdf_files:
                os.remove(os.path.join(txt_dir, txt_file))
                deleted_count += 1

        # Extraer PDFs nuevos
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath_pdfs, pdf_file)
            txt_path = os.path.join(txt_dir, pdf_file.replace(".pdf", ".txt"))

            if os.path.exists(txt_path):
                continue

            text = extract_txt(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Extracción de datos exitosa sobre \"{dirpath_pdfs}\\{pdf_file}\"")
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

