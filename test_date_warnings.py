#!/usr/bin/env python3
"""
Script de verificación para las correcciones de parsing de fechas
"""

import pandas as pd
import warnings

def test_date_parsing():
    """Probar que el parsing de fechas funcione sin advertencias"""
    print("🧪 Probando parsing de fechas...")

    # Capturar advertencias
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Datos de prueba en formato dd/mm/yyyy
        test_dates = ["15/08/2023", "01/01/2024", "31/12/2023", "07/04/2025"]

        # Crear DataFrame de prueba
        df_test = pd.DataFrame({"Fecha": test_dates})

        # Probar conversión con formato especificado
        s_fecha = pd.to_datetime(df_test["Fecha"], format="%d/%m/%Y", errors="coerce")

        # Verificar que no haya advertencias
        date_warnings = [warning for warning in w if "Parsing dates" in str(warning.message)]

        if date_warnings:
            print("❌ Aún hay advertencias de parsing de fechas:")
            for warning in date_warnings:
                print(f"   - {warning.message}")
            return False
        else:
            print("✅ No se encontraron advertencias de parsing de fechas")
            print(f"   Fechas convertidas correctamente: {len(s_fecha.dropna())} de {len(s_fecha)}")
            return True

def test_imports():
    """Verificar que los módulos se importen sin advertencias"""
    print("\n🔍 Probando importaciones...")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import UTILS.insights
            import UTILS.lineas_dashboard
            import UTILS.coples_dashboard
            import main

            # Filtrar solo advertencias de fecha
            date_warnings = [warning for warning in w if "Parsing dates" in str(warning.message)]

            if date_warnings:
                print("❌ Advertencias de fecha en importaciones:")
                for warning in date_warnings:
                    print(f"   - {warning.filename}:{warning.lineno}: {warning.message}")
                return False
            else:
                print("✅ Importaciones sin advertencias de fecha")
                return True

    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Verificación de correcciones de parsing de fechas")
    print("=" * 50)

    success = True
    success &= test_date_parsing()
    success &= test_imports()

    print("\n" + "=" * 50)
    if success:
        print("🎉 ¡Todas las verificaciones pasaron!")
        print("Las advertencias de parsing de fechas han sido corregidas.")
    else:
        print("⚠️  Algunas verificaciones fallaron.")
        exit(1)
