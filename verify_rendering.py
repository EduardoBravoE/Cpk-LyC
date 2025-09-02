#!/usr/bin/env python3
"""
Script de verificación para detectar problemas de renderizado duplicado
"""

import os
import re

def check_for_duplicate_renders():
    """Verificar si hay llamadas duplicadas a funciones de renderizado"""
    main_file = "main.py"

    if not os.path.exists(main_file):
        print(f"❌ Archivo {main_file} no encontrado")
        return False

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Buscar patrones de llamadas duplicadas
    patterns = [
        r'_render_lineas\(\)',
        r'_render_coples\(\)',
        r'_render_lineas_with_cache\(',
        r'_render_coples_with_cache\('
    ]

    duplicates_found = []

    for pattern in patterns:
        matches = re.findall(pattern, content)
        if len(matches) > 1:
            duplicates_found.append(f"{pattern}: {len(matches)} llamadas")

    if duplicates_found:
        print("❌ Se encontraron llamadas duplicadas:")
        for duplicate in duplicates_found:
            print(f"   - {duplicate}")
        return False
    else:
        print("✅ No se encontraron llamadas duplicadas a funciones de renderizado")
        return True

def check_main_structure():
    """Verificar la estructura general del main.py"""
    main_file = "main.py"

    if not os.path.exists(main_file):
        print(f"❌ Archivo {main_file} no encontrado")
        return False

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verificar que tenga la estructura correcta
    required_sections = [
        "def main():",
        "st.title",
        "selected_area_label = st.sidebar.radio",
        "if selected_area_domain == DOM_LINEAS:",
        "if selected_area_domain == DOM_COPLES:"
    ]

    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)

    if missing_sections:
        print("❌ Secciones faltantes en main.py:")
        for section in missing_sections:
            print(f"   - {section}")
        return False
    else:
        print("✅ Estructura de main.py correcta")
        return True

if __name__ == "__main__":
    print("🔍 Verificando problemas de renderizado duplicado...")
    print()

    success = True
    success &= check_for_duplicate_renders()
    success &= check_main_structure()

    print()
    if success:
        print("🎉 ¡Verificación completada! No se encontraron problemas de duplicación.")
    else:
        print("⚠️  Se encontraron problemas que deben corregirse.")
        exit(1)
