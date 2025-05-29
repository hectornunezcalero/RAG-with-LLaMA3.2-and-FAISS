import fitz  # PyMuPDF
import os
import re

def procesar_bloques(bloques):
    partes = []
    for bloque in bloques:
        contenido = ""
        if "lines" in bloque:
            for linea in bloque["lines"]:
                for span in linea["spans"]:
                    contenido += span["text"] + " "
            contenido = re.sub(r'\s+', ' ', contenido).strip()

        if contenido:
            partes.append(contenido)
    return "\n\n".join(partes)


def txt_extract(ruta_pdf):
    doc = fitz.open(ruta_pdf)
    texto_final = ""

    for pagina in doc:
        bloques = pagina.get_text("dict")["blocks"]
        if not bloques:
            continue

        ancho_pagina = pagina.rect.width
        media_x = ancho_pagina / 2

        # Clasificar bloques por columna
        izquierda = [b for b in bloques if b["bbox"][0] < media_x]
        derecha   = [b for b in bloques if b["bbox"][0] >= media_x]

        # Determinar si hay una o dos columnas
        total = len(izquierda) + len(derecha)
        porcentaje_izq = len(izquierda) / total if total > 0 else 1

        if 0.3 < porcentaje_izq < 0.7:
            # Doble columna: procesar por separado
            izquierda.sort(key=lambda b: b["bbox"][1])
            derecha.sort(key=lambda b: b["bbox"][1])
            texto_final += procesar_bloques(izquierda)
            texto_final += procesar_bloques(derecha)
        else:
            # Una columna: ordenar todo y procesar en bloque
            bloques.sort(key=lambda b: b["bbox"][1])
            texto_final += procesar_bloques(bloques)

        texto_final += "\n"

    return texto_final.strip()

def pdf_processor(data_dir="./pdfdata", output_dir="./txtdata"):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in os.listdir(data_dir):
        if archivo.endswith(".pdf"):
            ruta_pdf = os.path.join(data_dir, archivo)
            texto = txt_extract(ruta_pdf)

            salida = os.path.join(output_dir, archivo.replace(".pdf", ".txt"))
            with open(salida, "w", encoding="utf-8") as f:
                f.write(texto)

            print(f"Extracción de datos exitosa sobre: {archivo}")

if __name__ == "__main__":
    pdf_processor()

