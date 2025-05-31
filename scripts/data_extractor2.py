import fitz  # alias de PyMuPDF -> leer y extraer PDFs
import os # rutas, carpetas y archivos
import re # limpieza de texto mediante expresiones regulares


# Procesa los bloques de texto cada página del PDF, reduciendo espaciados innecesarios y soloexisitiendo los de separación de palabras y párrafos
def block_process(blocks):
    parts = []
    for block in blocks:
        content = ""
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    content += span["text"] + " "
            # reemplazar cualquier secuencia de uno o más espacios en blanco por un único espacio
            content = re.sub(r'\s+', ' ', content).strip()

        if content:
            parts.append(content)

    return "\n".join(parts)


# Extrae el texto de un PDF por bloques, teniendo en cuenta si cada hoja contiene una o dos columnas
def extract_txt(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"] # diccionario de información sobre la posición y contenido de cada bloque
        if not blocks:
            continue

        page_wide = page.rect.width
        # suma de las coordenadas 'x' de los bloques para determinar la posición media de la posible "línea imaginaria" (por si hay dos columnas no simetricas)
        average_x = sum(b["bbox"][0] for b in blocks) / len(blocks) if blocks else page_wide / 2

        left = [b for b in blocks if b["bbox"][0] < average_x]
        right = [b for b in blocks if b["bbox"][0] >= average_x]

        total = len(left) + len(right)
        left_percent = len(left) / total if total > 0 else 1

        # definir si la página tiene una o dos columnas, una vez comprobada el tamaño del lado izquierdo
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
def process_pdf(data_dir, output_dir):
    pdf_files = [file for file in os.listdir(data_dir) if file.endswith(".pdf")]

    if not pdf_files:
        print("No hay archivos PDF para extraer.")
        return

    updated = False
    for file in pdf_files:
        pdf_path = os.path.join(data_dir, file)
        txt_path = os.path.join(output_dir, file.replace(".pdf", ".txt"))

        if os.path.exists(txt_path):
            continue

        text = extract_txt(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Extracción de datos exitosa sobre: {file}")
        updated = True

    if not updated:
        print("No hay nada por extraer, todos los documentos están actualizados.")


# función principal
if __name__ == "__main__":
    process_pdf("./pdfdata", "./txtdata")

