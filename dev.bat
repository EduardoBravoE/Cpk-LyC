@echo off
REM Script de automatización para desarrollo Python/Streamlit
REM Ejecuta comandos comunes sin necesidad de confirmación

if "%1"=="run" (
    echo 🚀 Ejecutando aplicación Streamlit...
    cd /d "%~dp0"
    streamlit run main.py
    goto :eof
)

if "%1"=="test" (
    echo 🧪 Ejecutando pruebas...
    cd /d "%~dp0"
    python -m pytest tests/ -v --tb=short
    goto :eof
)

if "%1"=="lint" (
    echo 🔍 Ejecutando linter...
    cd /d "%~dp0"
    python -m flake8 . --max-line-length=100 --extend-ignore=E203,W503
    goto :eof
)

if "%1"=="build" (
    echo 📦 Construyendo ejecutable...
    cd /d "%~dp0"
    python -m PyInstaller main.spec
    goto :eof
)

if "%1"=="clean" (
    echo 🧹 Limpiando archivos temporales...
    cd /d "%~dp0"
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist __pycache__ rmdir /s /q __pycache__
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    goto :eof
)

if "%1"=="deps" (
    echo 📋 Verificando dependencias...
    cd /d "%~dp0"
    python -m pip check
    goto :eof
)

if "%1"=="help" (
    echo.
    echo 📚 Comandos disponibles:
    echo   run     - Ejecutar aplicación Streamlit
    echo   test    - Ejecutar pruebas
    echo   lint    - Ejecutar linter
    echo   build   - Construir ejecutable
    echo   clean   - Limpiar archivos temporales
    echo   deps    - Verificar dependencias
    echo   help    - Mostrar esta ayuda
    echo.
    echo 💡 Uso: dev.bat [comando]
    goto :eof
)

echo ❌ Comando no reconocido. Usa 'dev.bat help' para ver comandos disponibles.
goto :eof
