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
#           2.2 Tener en cuenta posibles Tablas/Esquemas                #
#           2.3 limpiarlos de espacios innecesarios.                    #
#        3. Guardar el texto extraído en archivos .txt en txtdata.      #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


import fitz  # alias de PyMuPDF -> extraer texto de archivos PDF
import os  # manejar rutas, directorios, archivos y operaciones del sistema de ficheros
import re  # limpiar y procesar texto mediante expresiones regulares
from collections import defaultdict # para contar ocurrencias de encabezados y pies de página

# Función para recolectar candidatos a encabezados y pies de página
def collect_header_footer_candidates(doc, top_frac=0.1, bottom_frac=0.1, min_pages_ratio=0.7):
    total_pages = len(doc)
    top_texts_count = defaultdict(int)
    bottom_texts_count = defaultdict(int)

    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            bbox = b["bbox"]
            y0, y1 = bbox[1], bbox[3]
            block_text = ""
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                block_text = block_text.strip()

            if not block_text:
                continue

            if y1 <= page_height * top_frac:
                top_texts_count[block_text] += 1
            if y0 >= page_height * (1 - bottom_frac):
                bottom_texts_count[block_text] += 1

    min_pages = int(total_pages * min_pages_ratio)
    header_texts = {text for text, count in top_texts_count.items() if count >= min_pages}
    footer_texts = {text for text, count in bottom_texts_count.items() if count >= min_pages}

    return header_texts, footer_texts


# Procesar los bloques de texto cada página del PDF, reduciendo espaciados innecesarios y existiendo solamente la separación de palabras y párrafos
def block_process(blocks, headers=None, footers=None):
    parts = []
    # Se eliminan dígitos para detectar encabezados/pies con números de página distintos
    normalized_headers = {re.sub(r'\d+', '', h.lower()).strip() for h in headers}
    normalized_footers = {re.sub(r'\d+', '', f.lower()).strip() for f in footers}
    # evaluación de cada bloque de texto
    for block in blocks:
        content = ""
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    content += span["text"] + " "
            # se reemplaza en el contenido cualquier secuencia de uno o más espacios en blanco por un único espacio
            content = re.sub(r'\s+', ' ', content).strip()

        normalized = re.sub(r'\d+', '', content.lower()).strip()
        if content and normalized not in normalized_headers and normalized not in normalized_footers:
            parts.append(content)
    return "\n".join(parts)


# Clasificar los bloques en texto corrido (columnas) y no texto (esquemas, diagramas o tablas)
def classify_blocks(blocks, page_width):
    text_blocks = []
    non_text_blocks_col = []
    non_text_blocks_full = []

    for block in blocks:
        num_lines = len(block.get("lines", []))
        num_spans = sum(len(line.get("spans", [])) for line in block.get("lines", []))

        # Consideramos bloque texto si tiene densidad
        if num_lines >= 1 and num_spans >= 3:
            block["__type__"] = "text"
            text_blocks.append(block)
        else:
            # No textual: decidir si full-width o columna
            block_width = block["bbox"][2] - block["bbox"][0]
            if block_width > page_width * 0.8:  # umbral 80% del ancho página
                block["__type__"] = "nontext_full"
                non_text_blocks_full.append(block)
            else:
                block["__type__"] = "nontext_col"
                non_text_blocks_col.append(block)

    return text_blocks, non_text_blocks_col, non_text_blocks_full


# Determinar si una página tiene una o dos columnas, separando los bloques de texto consecuentemente
def is_two_column_layout(blocks, page_width):
    if not blocks:
        return False, [], []

    centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]
    centers_sorted = sorted(centers)

    max_gap = 0
    split_pos = None
    for i in range(len(centers_sorted) - 1):
        gap = centers_sorted[i + 1] - centers_sorted[i]
        if gap > max_gap:
            max_gap = gap
            split_pos = (centers_sorted[i + 1] + centers_sorted[i]) / 2

    if max_gap < page_width * 0.15:
        return False, blocks, []

    left = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < split_pos]
    right = [b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= split_pos]

    total = len(blocks)
    left_ratio = len(left) / total
    right_ratio = len(right) / total

    if left_ratio > 0.25 and right_ratio > 0.25:
        return True, left, right
    else:
        return False, blocks, []


# Ordenar los bloques de texto verticalmente
def sort_blocks_visually(blocks):
    # se ordenan los bloques por su coordenada vertical 'y' (de arriba a abajo)
    return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))


# Filtrar encabezados y pies de página basándose en su posición vertical
def filter_header_footer(blocks, page_height, margin_ratio=0.05):
    margin = page_height * margin_ratio
    return [
        b for b in blocks
        if b["bbox"][1] >= margin and b["bbox"][3] <= page_height - margin
    ]


# Extraer el texto de un PDF por bloques, clasificando por tipo de bloque (texto o esquemas/tablas)
def extract_txt(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = ""

    header_texts, footer_texts = collect_header_footer_candidates(doc)

    for page in doc:
        blocks = page.get_text("dict")["blocks"]  # diccionario de información sobre los bloques de información: posición y líneas con spans de cada bloque de información
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

        # se clasifican los bloques entre texto corrido (columnas) y otros elementos (esquemas, tablas, etc.)
        text_blocks, nontext_col_blocks, nontext_full_blocks = classify_blocks(blocks, page_width)

        # Quitar encabezados/pies por ubicación
        text_blocks = filter_header_footer(text_blocks, page_height)

        # Clasificación por columnas
        is_two_col, left_blocks, right_blocks = is_two_column_layout(text_blocks, page_width)

        for b in left_blocks:
            b["__col__"] = "left"
        for b in right_blocks:
            b["__col__"] = "right"

        left_min_x = min([b["bbox"][0] for b in left_blocks], default=0)
        left_max_x = max([b["bbox"][2] for b in left_blocks], default=0)
        right_min_x = min([b["bbox"][0] for b in right_blocks], default=0)
        right_max_x = max([b["bbox"][2] for b in right_blocks], default=0)

        left_nontext = [b for b in nontext_col_blocks if left_min_x <= b["bbox"][0] <= left_max_x]
        right_nontext = [b for b in nontext_col_blocks if right_min_x <= b["bbox"][0] <= right_max_x]

        all_blocks_left = sort_blocks_visually(left_blocks + left_nontext)
        all_blocks_right = sort_blocks_visually(right_blocks + right_nontext)
        all_blocks_full = sort_blocks_visually(nontext_full_blocks)

        # Orden de lectura: izquierda ➝ derecha ➝ ancho completo
        page_text = ""
        for block in all_blocks_left:
            page_text += block_process([block], header_texts, footer_texts) + "\n"
        for block in all_blocks_right:
            page_text += block_process([block], header_texts, footer_texts) + "\n"
        for block in all_blocks_full:
            # Consideramos tabla si tiene múltiples líneas y múltiples spans
            num_lines = len(block.get("lines", []))
            num_spans = sum(len(line.get("spans", [])) for line in block.get("lines", []))

            # Se obtiene el texto limpio del bloque una vez
            raw_text = block_process([block], header_texts, footer_texts)
            lower_text = raw_text.lower()

            keywords = ["table", "tables", "tab."]

            # Si es un bloque de texto con el nombre de la tabla, lo incluimos
            if raw_text and any(kw in lower_text for kw in keywords):
                page_text += raw_text + "\n"
            # Si es una Tabla con múltiples líneas y spans, lo incluimos
            elif num_lines >= 2 and num_spans >= 4:
                page_text += raw_text + "\n"
            # De lo contrario lo ignoramos
            else:
                pass  # esquema sin interés textual

        final_text += page_text.strip() + "\n"

    return final_text.strip()


# Extraer el texto de los PDFs y guardarlos como archivos de texto
def process_pdf(pdf_root, txt_root):
    extracted_count = 0
    deleted_count = 0
    nopdf_count = 0

    # se recorre el directorio raíz que contiene los subdirectorios de PDFs
    print("Recorriendo directorio de PDFs y comprobando versiones en texto...")
    for dirpath_pdfs, _, files in os.walk(pdf_root):

        if dirpath_pdfs == pdf_root:
            continue

        pdf_files = [f for f in files if f.endswith(".pdf")]

        if not pdf_files:
            nopdf_count += 1
            continue

        # se usará la misma subestructura en pdf_data que en txt_data
        rel_path = os.path.relpath(dirpath_pdfs, pdf_root)  # ej: \Papers 20-21 (no se incluye el \pdf_data)
        txt_dir = os.path.join(txt_root, rel_path)  # ej: .\txt_data\Papers 20-21

        # si no estaban creados los subdirectorios de texto, se crean
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # se eliminan los .txt cuyos pdfs ya no existen, si es que hay .txts
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith(".txt") and txt_file.replace(".txt", ".pdf") not in pdf_files:
                os.remove(os.path.join(txt_dir, txt_file))
                deleted_count += 1

        # se extraen los PDFs nuevos
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath_pdfs, pdf_file)  # \ej: pdf_data\Papers 20-21\articulo1.pdf
            txt_path = os.path.join(txt_dir, pdf_file.replace(".pdf", ".txt"))  # \ej: .\txt_data\Papers 20-21\articulo1.txt

            # si el archivo de texto ya ha sido creado en ejecuciones anteriores (y no es un archivo vacío por error), se omite la extracción
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                continue

            # se llama al extractor de texto para ese PDF para su posterior escritura y almacenamiento en su versión .txt
            text = extract_txt(pdf_path)
            if text.strip():
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"Extracción de datos exitosa sobre \"{pdf_path}\"")
                extracted_count += 1
            else:
                print(f"Advertencia: {pdf_path} no tiene bloques útiles o legibles.")

    if deleted_count > 0:
        print(f"Se han eliminado {deleted_count} archivos de texto correspondientes a PDFs que ya no existen.")

    if extracted_count > 0:
        print(f"Total de archivos extraídos a texto en esta ejecución: {extracted_count}")

    if extracted_count == 0 and deleted_count == 0:
        if nopdf_count > 0:
            print(f"No se han encontrado PDFs en la(s) {nopdf_count} carpeta(s).")
        else:
            print("No quedan PDFs por extraer, todos los archivos ya tienen su versión .txt.")


# Función principal
if __name__ == "__main__":
    process_pdf("./pdfdata", "./txtdata")

