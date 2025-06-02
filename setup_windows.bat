@echo off
setlocal

REM Verificar Python
where python >nul 2>nul
if %errorlevel% NEQ 0 (
    echo Python no está instalado.
    echo Descárgalo en: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar pip
where pip >nul 2>nul
if %errorlevel% NEQ 0 (
    echo pip no está instalado.
    echo Asegúrate de haber marcado "Add Python to PATH" al instalar.
    pause
    exit /b 1
)

REM Crear entorno virtual
if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Instalar dependencias
if exist requirements.txt (
    echo Instalando dependencias...
    pip install -r requirements.txt
) else (
    echo No se encontró requirements.txt
)

echo Entorno configurado correctamente.
pause
