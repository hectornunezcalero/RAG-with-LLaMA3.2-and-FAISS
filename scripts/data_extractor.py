import fitz  # PyMuPDF
import os

# extracción del texto de los archivos PDF
def txt_extract(ruta_pdf):
    doc = fitz.open(ruta_pdf)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# parseo de pdf y conversión pdf -> txt
def pdf_processor(data_dir="./pdfdata", output_dir="./txtdata"):
    for archivo in os.listdir(data_dir):
        if archivo.endswith(".pdf"):
            ruta_pdf = os.path.join(data_dir, archivo) # construcción de ruta del pdf independientemente del so
            texto = txt_extract(ruta_pdf)

            salida = os.path.join(output_dir, archivo.replace(".pdf", ".txt")) # construcción de ruta del txt sobre su pdf correspondiente
            with open(salida, "w", encoding="utf-8") as f:
                f.write(texto) # escritura del texto extraído en el archivo txt
            print(f"Extracción de datos exitosa sobre: {archivo}")

if __name__ == "__main__":
    pdf_processor()
