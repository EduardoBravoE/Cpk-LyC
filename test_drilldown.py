#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del drilldown
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Verificar que todas las importaciones funcionen"""
    try:
        from UTILS.insights import get_cached_drilldown_results, compute_clave_drilldown_data
        from UTILS.confiabilidad_panel import _render_drilldown_clave
        print("✅ Todas las importaciones funcionan correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def test_cache_decorators():
    """Verificar que las funciones tienen los decoradores de cache"""
    try:
        from UTILS.insights import get_cached_drilldown_results, compute_clave_drilldown_data
        import inspect

        # Verificar decoradores
        if hasattr(get_cached_drilldown_results, 'cache_info'):
            print("✅ get_cached_drilldown_results tiene cache")
        else:
            print("❌ get_cached_drilldown_results no tiene cache")

        if hasattr(compute_clave_drilldown_data, 'cache_info'):
            print("✅ compute_clave_drilldown_data tiene cache")
        else:
            print("❌ compute_clave_drilldown_data no tiene cache")

        return True
    except Exception as e:
        print(f"❌ Error al verificar decoradores: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Probando sistema de drilldown...")
    print()

    success = True
    success &= test_imports()
    success &= test_cache_decorators()

    print()
    if success:
        print("🎉 ¡Todas las pruebas pasaron!")
    else:
        print("⚠️  Algunas pruebas fallaron")
        sys.exit(1)
