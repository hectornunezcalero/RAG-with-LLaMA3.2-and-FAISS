#!/bin/bash
set -e

# --- Verificación Python ---
if ! command -v python3 &> /dev/null; then
    echo "Python3 no está instalado."
    echo "Instalalo con:"
    echo "  - Debian/Ubuntu: sudo apt install python3"
    echo "  - Fedora:        sudo dnf install python3"
    echo "  - macOS:         brew install python"
    exit 1
fi

# --- Verificación pip ---
if ! command -v pip3 &> /dev/null; then
    echo "pip3 no está instalado."
    echo "Instálalo con:"
    echo "  - Debian/Ubuntu: sudo apt install python3-pip"
    echo "  - Fedora:        sudo dnf install python3-pip"
    echo "  - macOS:         brew install pip"
    exit 1
fi

# --- Crear entorno virtual si no existe ---
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# --- Activar entorno virtual ---
echo "Activando entorno virtual..."
source venv/bin/activate

# --- Instalar dependencias ---
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias..."
    pip install -r requirements.txt
else
    echo "No se encontró el archivo requirements.txt"
fi

echo "Entorno configurado correctamente."
