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
#        1. Extraer texto de los archivos PDF de pdfdata además de:     #
#           1.1 Estructurarlos en una o dos columnas (o híbrido).       #
#           1.2 Limpiarlos de espacios y contenido innecesario.         #
#        2. Guardar el texto extraído en archivos .txt en txtdata.      #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

import fitz  # alias de PyMuPDF -> extraer texto de archivos PDF
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import re  # limpiar y procesar texto mediante expresiones regulares

PDF_ROOT_PATH = ".\\pdfdata"
TXT_ROOT_PATH = ".\\txtdata"


# Procesar los bloques de información de cada la página
def block_process(blocks):
    parts = []
    for block in blocks:
        content = ""
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if text:
                        content += text + " "
            # se reducen espaciados innecesarios, existiendo solamente la separación de palabras y párrafos
            content = re.sub(r'\s+', ' ', content).strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


# Determinar si los bloques de texto están organizados en una o dos columnas
def is_two_column_layout(blocks, page_width):
    # se usan las posiciones horizontales de los bloques para determinar si hay dos columnas
    centers = []
    for b in blocks:
        centers.append((b["bbox"][0] + b["bbox"][2]) / 2)
    centers_sorted = sorted(centers)
    max_gap = 0
    split_pos = None

    # se busca la mayor diferencia de espacio horizontal entre bloques consecutivos para tomarla como posible separación de columnas
    for i in range(len(centers_sorted) - 1):
        gap = centers_sorted[i + 1] - centers_sorted[i]
        if gap > max_gap:
            max_gap = gap
            split_pos = (centers_sorted[i + 1] + centers_sorted[i]) / 2

    # si la mayor separación entre bloques es menor que el 15% del ancho de la página, se asume que hay una única columna
    if max_gap < page_width * 0.15:
        return blocks, []

    # si no, se separan los bloques en dos grupos: bloques de la columna izquierda y bloques de la columna derecha
    left = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < split_pos]
    right = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= split_pos]

    # de seguido, se comprueba si ambos grupos tienen al menos un 25% de los bloques totales
    total = len(blocks)
    left_ratio = len(left) / total
    right_ratio = len(right) / total
    if left_ratio > 0.20 and right_ratio > 0.20:
        return left, right
    #si no, ese bloque que ocupa menos del 20% se considera de la misma columna que el otro
    else:
        return blocks, []


# Analizar el formato de columnas dentro de una zona vertical
def analyze_zone_layout(zone_blocks, page_width):
    left, right = is_two_column_layout(zone_blocks, page_width)
    if right:
        return True, (left, right)
    else:
        return False, zone_blocks


# Agrupar bloques en zonas verticales según su proximidad en Y
def cluster_blocks_vertically(blocks):
    # se ordenan y agrupan los bloques por su posición vertical, para así agrupar bloques cercanos entre sí
    blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][1])
    clusters = []
    current_cluster = [blocks_sorted[0]]
    current_max_y = blocks_sorted[0]["bbox"][3]
    # se establece la máxima distancia vertical entre bloques consecutivos para considerarlos parte del mismo grupo
    max_gap = 15

    # se valorara la distancia entre bloques consecutivos (hueco final-comienzo entre bloques)
    for b in blocks_sorted[1:]:
        y_top = b["bbox"][1]
        # si se cumple la distancia, se añade el bloque al grupo actual
        if y_top - current_max_y <= max_gap:
            current_cluster.append(b)
            # se actualiza la posición máxima del bloque actual, por si se ha añadido un bloque más bajo
            current_max_y = max(current_max_y, b["bbox"][3])
        # si no, se cierra el grupo actual y se inicia uno nuevo
        else:
            clusters.append(current_cluster)
            current_cluster = [b]
            current_max_y = b["bbox"][3]
    # se asegura el añadir el último grupo de bloques
    clusters.append(current_cluster)
    return clusters


# Filtrar encabezados y pies de página (estrictamente marginados por estadística previa) de los bloques de texto
def filter_header_footer(blocks, page_height):
    margin_top = page_height * 0.08
    margin_bot = page_height * 0.925

    not_headfoot_blocks = [b for b in blocks if not (b["bbox"][3] < margin_top or b["bbox"][1] > margin_bot)]
    return not_headfoot_blocks


# Extraer el texto de un PDF por bloques, clasificando por tipo de bloque (texto o esquemas)
def extract_txt(pdf_path):
    doc = fitz.open(pdf_path)
    accumulated_lines = []

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

        page_width = page.rect.width
        page_height = page.rect.height

        # se eliminan los encabezados y pie de página
        nonheadfoot_blocks = filter_header_footer(blocks, page_height)

        # Agrupamos todos los bloques en zonas verticales
        vertical_clusters = cluster_blocks_vertically(nonheadfoot_blocks)

        page_text = ""

        # atendiendo los clusters ordenados, iremos formando el texto de la página
        for zone in vertical_clusters:
            two_col, content = analyze_zone_layout(zone, page_width)

            # información en dos columnas
            if two_col is True:
                left_blocks, right_blocks = content
                left_sorted = sorted(left_blocks, key=lambda b: b["bbox"][1])
                right_sorted = sorted(right_blocks, key=lambda b: b["bbox"][1])

                for block in left_sorted:
                    page_text += block_process([block]) + "\n"
                for block in right_sorted:
                    page_text += block_process([block]) + "\n"

            # información en una sola columna
            elif two_col is False:
                sorted_blocks = sorted(content, key=lambda b: b["bbox"][1])
                for block in sorted_blocks:
                    page_text += block_process([block]) + "\n"

        # se acumula el texto de la página
        accumulated_lines.append(page_text.strip())

    # se une el texto de todas las páginas
    accumulated_text = "\n".join(accumulated_lines)

    # se limpian las líneas de posibles encabezados/pies de figuras no capturadas
    cleaned_lines = []
    for line in accumulated_text.splitlines():
        if re.match(r'^(Fig\.|Figure)\s*\d+[\.:]?\s+.*$', line.strip(), re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    # se crea la cadena de texto casi final del PDF mediante la lista de cadenas (antes de eliminar las referencias innecesarias)
    semifinal_text = "\n".join(cleaned_lines)

    # si existen referencias al final del pdf, se eliminan del texto útil
    match = re.search(r'^References\b', semifinal_text, re.MULTILINE)
    if match:
        final_text = semifinal_text[:match.start()]
    else:
        final_text = semifinal_text

    return final_text.strip()


# Procesar los PDFs y extraer su texto a archivos .txt
def process_pdf(pdf_root: str, txt_root: str):
    nopdf_count = 0
    deleted_count = 0
    extracted_count = 0
    notextracted_count = 0

    # se recorre el directorio raíz que contiene los subdirectorios de PDFs
    print("Recorriendo directorio raíz de PDFs y comprobando versiones en texto...")
    for pdfs_dirpath, _, files in os.walk(pdf_root):

        # se omite el escaneo del directorio raíz de PDFs
        if pdfs_dirpath == pdf_root:
            continue

        pdf_files = [f for f in files if f.endswith(".pdf")]
        rel_path = os.path.relpath(pdfs_dirpath, pdf_root) # ej.: \Papers 20-21 (no se incluye el directorio raíz)
        txt_dir = os.path.join(txt_root, rel_path) # ej.: .\txt_data\Papers 20-21

        # si no hay PDFs, se eliminan todos los .txt
        if not pdf_files:
            if os.path.exists(txt_dir):
                for txt_file in os.listdir(txt_dir):
                    if txt_file.endswith(".txt"):
                        os.remove(os.path.join(txt_dir, txt_file))
                        deleted_count += 1
            nopdf_count += 1
            continue

        # si no estaban ya creados los subdirectorios de texto, se crean
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # se recogen los archivos .txt del el directorio actual
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]

        # se crea un conjunto de archivos .txt esperados basados en los PDFs
        expected_txt_files = {pdf_file.replace(".pdf", ".txt") for pdf_file in pdf_files}

        # se eliminan los .txt cuyos pdfs ya no existen entre los que sí hay
        for txt_file in txt_files:
            if txt_file not in expected_txt_files:
                os.remove(os.path.join(txt_dir, txt_file))
                deleted_count += 1

        # se crean las rutas completas de los archivos .txt
        for pdf_file in pdf_files:
            txt_file = pdf_file.replace(".pdf", ".txt")
            txt_path = os.path.join(txt_dir, txt_file)

            # si el archivo de texto ya ha sido creado en ejecuciones anteriores (y no es un archivo vacío por error), se omite la extracción
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                continue

            # se llama al extractor de texto con el pdf actual para su posterior escritura y almacenamiento de su versión .txt
            pdf_path = os.path.join(pdfs_dirpath, pdf_file)
            text = extract_txt(pdf_path)

            if text:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Extracción de datos exitosa sobre \"{pdf_path}\".")
                extracted_count += 1

            else:
                print(f"Advertencia: {pdf_path} no tiene bloques útiles o legibles.")
                notextracted_count += 1

    if deleted_count > 0:
        print(f"Se ha(n) eliminado {deleted_count} archivos de texto correspondientes a PDFs que ya no existen.")

    if extracted_count > 0:
        print(f"Total de archivos extraídos a texto en esta ejecución: {extracted_count}")

    if notextracted_count > 0:
        print(f"Total de archivos no extraídos por no contener bloques útiles o legibles: {notextracted_count}.")

    if extracted_count == 0 and deleted_count == 0:
        if nopdf_count > 0:
            print(f"No se han encontrado PDFs en la(s) {nopdf_count} carpeta(s).")
        else:
            print("No quedan PDFs por extraer, todos los archivos ya tienen su versión .txt.")


# Función principal
if __name__ == "__main__":
    process_pdf(PDF_ROOT_PATH, TXT_ROOT_PATH)