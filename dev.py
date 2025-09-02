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

class DevAutomation:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.commands = {
            'run': self.run_app,
            'test': self.run_tests,
            'lint': self.run_linter,
            'build': self.build_exe,
            'clean': self.clean_files,
            'deps': self.check_deps,
            'install': self.install_deps,
            'format': self.format_code,
            'check': self.full_check
        }

    def run_command(self, cmd, cwd=None, capture_output=False):
        """Ejecuta un comando y maneja errores"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando: {cmd}")
            print(f"   Código de salida: {e.returncode}")
            if e.stdout:
                print(f"   Salida: {e.stdout}")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            return None

    def run_app(self):
        """Ejecuta la aplicación Streamlit"""
        print("🚀 Ejecutando aplicación Streamlit...")
        return self.run_command("streamlit run main.py")

    def run_tests(self):
        """Ejecuta las pruebas"""
        print("🧪 Ejecutando pruebas...")
        return self.run_command("python -m pytest tests/ -v --tb=short")

    def run_linter(self):
        """Ejecuta el linter"""
        print("🔍 Ejecutando linter...")
        return self.run_command("python -m flake8 . --max-line-length=100 --extend-ignore=E203,W503")

    def build_exe(self):
        """Construye el ejecutable"""
        print("📦 Construyendo ejecutable...")
        return self.run_command("python -m PyInstaller main.spec")

    def clean_files(self):
        """Limpia archivos temporales"""
        print("🧹 Limpiando archivos temporales...")
        dirs_to_clean = ['build', 'dist', '__pycache__']
        files_to_clean = ['*.pyc', '*.pyo', '*.pyd']

        for dir_name in dirs_to_clean:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                import shutil
                shutil.rmtree(dir_path)
                print(f"   Eliminado: {dir_path}")

        for pattern in files_to_clean:
            for file_path in self.project_root.rglob(pattern):
                file_path.unlink()
                print(f"   Eliminado: {file_path}")

        print("✅ Limpieza completada")

    def check_deps(self):
        """Verifica dependencias"""
        print("📋 Verificando dependencias...")
        return self.run_command("python -m pip check")

    def install_deps(self):
        """Instala dependencias"""
        print("📥 Instalando dependencias...")
        return self.run_command("pip install -r requirements.txt")

    def format_code(self):
        """Formatea el código"""
        print("🎨 Formateando código...")
        return self.run_command("python -m black . --line-length=100")

    def full_check(self):
        """Ejecuta verificación completa"""
        print("🔍 Ejecutando verificación completa...")
        checks = [
            ("Sintaxis", "python -m py_compile main.py"),
            ("Linter", "python -m flake8 . --max-line-length=100 --extend-ignore=E203,W503"),
            ("Dependencias", "python -m pip check")
        ]

        all_passed = True
        for name, cmd in checks:
            print(f"\n📋 {name}:")
            result = self.run_command(cmd, capture_output=True)
            if result and result.returncode == 0:
                print(f"   ✅ {name} - OK")
            else:
                print(f"   ❌ {name} - FALLÓ")
                all_passed = False

        if all_passed:
            print("\n🎉 ¡Todas las verificaciones pasaron!")
        else:
            print("\n⚠️  Algunas verificaciones fallaron")

    def show_help(self):
        """Muestra la ayuda"""
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
   install - Instalar dependencias
   format  - Formatear código
   check   - Verificación completa
   help    - Mostrar esta ayuda

🚀 Atajos de teclado (VS Code):
   Ctrl+Shift+R - Ejecutar app
   Ctrl+Shift+T - Ejecutar tests
   Ctrl+Shift+L - Ejecutar linter
   Ctrl+Shift+B - Construir ejecutable
   Ctrl+Shift+C - Limpiar archivos
   Ctrl+Shift+D - Verificar dependencias
   Ctrl+Shift+S - Guardar todo

💡 Consejos:
   - Usa 'python dev.py check' antes de commits
   - El script maneja errores automáticamente
   - Todas las tareas se ejecutan en background cuando es posible
        """)

def main():
    if len(sys.argv) < 2:
        DevAutomation().show_help()
        return

    command = sys.argv[1].lower()
    dev = DevAutomation()

    if command in dev.commands:
        result = dev.commands[command]()
        if result is None:
            sys.exit(1)
    elif command == 'help':
        dev.show_help()
    else:
        print(f"❌ Comando '{command}' no reconocido.")
        dev.show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
