import fitz  # PyMuPDF
import os

def extraer_texto(ruta_pdf):
    doc = fitz.open(ruta_pdf)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

def procesar_todos(data_dir="./data", output_dir="./data/textos"):
    os.makedirs(output_dir, exist_ok=True)
    for archivo in os.listdir(data_dir):
        if archivo.endswith(".pdf"):
            ruta_pdf = os.path.join(data_dir, archivo)
            texto = extraer_texto(ruta_pdf)
            salida = os.path.join(output_dir, archivo.replace(".pdf", ".txt"))
            with open(salida, "w", encoding="utf-8") as f:
                f.write(texto)
            print(f"[✓] Texto extraído de: {archivo}")

if __name__ == "__main__":
    procesar_todos()
