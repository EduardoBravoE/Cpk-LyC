# Proyecto Cpk's

## Descripción
Aplicación en Streamlit para identificar 'Productos Críticos' (combinaciones Diámetro-Libraje-Acero-Rosca) candidatos a análisis Cpk. Prioriza productos con alto índice de rechazo sin calcular Cpk directamente.

## Instalación
1. Clona el repositorio.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta: `streamlit run main.py`.

## Estructura del Proyecto
- `DATOS/`: Archivos Excel de datos.
- `REFERENCIAS/`: Catálogo de claves y archivos opcionales (overrides, patterns, locks).
- `UTILS/`: Módulos de lógica (common.py, insights.py, dashboards).
- `main.py`: Punto de entrada.

## Uso
- Selecciona dominio (Líneas o Coples).
- Aplica filtros y visualiza productos críticos.
- Usa el panel de confiabilidad para análisis de mapeo.

## Archivos Opcionales
- `overrides_rechazos.csv`: Sobrescribir mapeos manuales.
- `clave_patterns.csv`: Patrones de coincidencia por clave.
- `mapping_lock.csv`: Bloquear mapeos específicos.

## Autor
Eduardo BRAVO
