# Verificar Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python no está instalado."
    Write-Host "Descárgalo desde: https://www.python.org/downloads/"
    exit 1
}

# Verificar pip
if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Host "pip no está instalado."
    Write-Host "Asegúrate de haber añadido Python al PATH."
    exit 1
}

# Crear entorno virtual
if (-not (Test-Path "venv")) {
    Write-Host "Creando entorno virtual..."
    python -m venv venv
}

# Activar entorno virtual
Write-Host "Activando entorno virtual..."
& venv\Scripts\Activate.ps1

# Instalar dependencias
if (Test-Path "requirements.txt") {
    Write-Host "Instalando dependencias..."
    pip install -r requirements.txt
} else {
    Write-Host "No se encontró requirements.txt"
}

Write-Host "Entorno configurado correctamente."
