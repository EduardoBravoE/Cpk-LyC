# 🚀 Sistema de Automatización de Desarrollo

Este proyecto incluye un sistema completo de automatización para agilizar tu flujo de trabajo con VS Code y GitHub Copilot.

## ⚡ Inicio Rápido

### Opción 1: Script Python (Recomendado)
```bash
# Ejecutar la aplicación
python dev.py run

# Verificación completa antes de commit
python dev.py check

# Limpiar y reconstruir
python dev.py clean && python dev.py build
```

### Opción 2: Script Batch (Windows)
```cmd
# Ejecutar la aplicación
dev.bat run

# Verificación completa
dev.bat deps
```

### Opción 3: Tareas de VS Code
- Presiona `Ctrl+Shift+P` → "Tasks: Run Task"
- Selecciona la tarea deseada

## 🎹 Atajos de Teclado

| Atajo | Acción |
|-------|--------|
| `Ctrl+Shift+R` | 🚀 Ejecutar Streamlit |
| `Ctrl+Shift+T` | 🧪 Ejecutar tests |
| `Ctrl+Shift+L` | 🔍 Ejecutar linter |
| `Ctrl+Shift+B` | 📦 Construir ejecutable |
| `Ctrl+Shift+C` | 🧹 Limpiar archivos |
| `Ctrl+Shift+D` | 📋 Verificar dependencias |
| `Ctrl+Shift+S` | 💾 Guardar todo |

## 📋 Comandos Disponibles

### Script Python (`dev.py`)
```bash
python dev.py run      # Ejecutar aplicación
python dev.py test     # Ejecutar pruebas
python dev.py lint     # Ejecutar linter
python dev.py build    # Construir ejecutable
python dev.py clean    # Limpiar temporales
python dev.py deps     # Verificar dependencias
python dev.py install  # Instalar dependencias
python dev.py format   # Formatear código
python dev.py check    # Verificación completa
python dev.py help     # Mostrar ayuda
```

### Script Batch (`dev.bat`)
```cmd
dev.bat run      # Ejecutar aplicación
dev.bat test     # Ejecutar pruebas
dev.bat lint     # Ejecutar linter
dev.bat build    # Construir ejecutable
dev.bat clean    # Limpiar temporales
dev.bat deps     # Verificar dependencias
dev.bat help     # Mostrar ayuda
```

## 🔧 Configuraciones Optimizadas

### VS Code Settings (`.vscode/settings.json`)
- ✅ Autocompletado agresivo
- ✅ Guardado automático
- ✅ Terminal sin confirmaciones
- ✅ Git autofetch activado

### Tareas Automatizadas (`.vscode/tasks.json`)
- ✅ Ejecución en background cuando es posible
- ✅ Output compartido en panel
- ✅ Sin prompts de confirmación

### Atajos Personalizados (`.vscode/keybindings.json`)
- ✅ Acceso rápido a tareas comunes
- ✅ Navegación fluida

## 🎯 Flujo de Trabajo Optimizado

### Desarrollo Diario
1. **Abre el proyecto** → VS Code se configura automáticamente
2. **Presiona `Ctrl+Shift+R`** → App ejecutándose
3. **Edita código** → Autoguardado activado
4. **Presiona `Ctrl+Shift+T`** → Tests ejecutándose
5. **Presiona `Ctrl+Shift+L`** → Linter verificando

### Antes de Commit
```bash
python dev.py check
```
Esto ejecuta:
- ✅ Verificación de sintaxis
- ✅ Linter (flake8)
- ✅ Verificación de dependencias

### Build y Distribución
```bash
python dev.py clean && python dev.py build
```

## 🚀 Beneficios

### Velocidad
- ⚡ Ejecución instantánea de comandos comunes
- ⚡ Sin esperas por confirmaciones
- ⚡ Autocompletado inteligente

### Productividad
- 🎯 Atajos de un solo paso
- 🎯 Scripts que hacen múltiples tareas
- 🎯 Configuración automática

### Confiabilidad
- 🛡️ Verificaciones automáticas
- 🛡️ Manejo de errores integrado
- 🛡️ Limpieza automática

## 📝 Notas Importantes

- Los scripts están diseñados para Windows (PowerShell)
- Todas las tareas se ejecutan en el directorio del proyecto
- Los errores se manejan automáticamente con mensajes informativos
- El sistema es extensible - puedes agregar más comandos fácilmente

## 🔧 Personalización

### Agregar Nuevo Comando
1. Edita `dev.py` y agrega el método en la clase `DevAutomation`
2. Actualiza el diccionario `self.commands`
3. Agrega el comando correspondiente en `dev.bat`
4. Crea una tarea en `.vscode/tasks.json`
5. Asigna un atajo en `.vscode/keybindings.json`

¡Tu flujo de trabajo ahora es completamente automatizado! 🎉
