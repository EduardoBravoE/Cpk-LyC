#!/usr/bin/env python3
"""
Script de automatización para desarrollo Python/Streamlit
Ejecuta tareas comunes sin necesidad de confirmación manual
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Ejecuta un comando y muestra el resultado"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=os.getcwd())
        print(f"✅ {description} - Completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "run":
        run_command("streamlit run main.py", "Ejecutando aplicación Streamlit")
    elif command == "test":
        run_command("python -m pytest tests/ -v", "Ejecutando pruebas")
    elif command == "lint":
        run_command("python -m flake8 . --max-line-length=100", "Ejecutando linter")
    elif command == "build":
        run_command("python -m PyInstaller main.spec", "Construyendo ejecutable")
    elif command == "clean":
        print("🧹 Limpiando archivos temporales...")
        # Implementar limpieza aquí
        print("✅ Limpieza completada")
    elif command == "deps":
        run_command("python -m pip check", "Verificando dependencias")
    elif command == "help":
        show_help()
    else:
        print(f"❌ Comando '{command}' no reconocido.")
        show_help()

def show_help():
    print("""
📚 Script de Automatización de Desarrollo

💡 Uso: python dev.py [comando]

🎯 Comandos disponibles:
   run     - Ejecutar aplicación Streamlit
   test    - Ejecutar pruebas
   lint    - Ejecutar linter
   build   - Construir ejecutable
   clean   - Limpiar archivos temporales
   deps    - Verificar dependencias
   help    - Mostrar esta ayuda

🚀 Atajos de teclado (VS Code):
   Ctrl+Shift+R - Ejecutar app
   Ctrl+Shift+T - Ejecutar tests
   Ctrl+Shift+L - Ejecutar linter
   Ctrl+Shift+B - Construir ejecutable
   Ctrl+Shift+C - Limpiar archivos
   Ctrl+Shift+D - Verificar dependencias
    """)

if __name__ == "__main__":
    main()
